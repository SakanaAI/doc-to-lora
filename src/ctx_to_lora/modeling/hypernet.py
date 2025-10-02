import logging
from collections.abc import Iterable
from dataclasses import dataclass
from functools import partial
from math import sqrt
from typing import Any

import torch
from einops import unpack
from einops.layers.torch import EinMix as Mix
from jaxtyping import Float, Integer
from peft import (
    LoraConfig,
    LoraRuntimeConfig,
    PeftConfig,
    PeftModel,
)
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import PeftType, TaskType
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel

from ctx_to_lora.configs import (
    AggregatorArguments,
    CtxEncoderArguments,
    HypernetArguments,
)
from ctx_to_lora.data.processing import tokenize_ctx_text
from ctx_to_lora.model_loading import (
    get_model,
    get_tokenizer,
)
from ctx_to_lora.modeling.aggregator import (
    AGGREGATOR_CLS,
    AggregatorConfig,
    get_aggregator_config,
)
from ctx_to_lora.modeling.ctx_encoder import CTX_ENCODER_CLS, CTX_ENCODER_TYPE
from ctx_to_lora.modeling.lora_layer import (
    apply_lora_to_layers,
    lora_forward,
    lora_forward_packed,
)
from ctx_to_lora.modeling.lora_merger import combine_lora
from ctx_to_lora.utils import (
    get_layers,
    get_num_layers,
    get_peft_in_out_features,
    get_peft_modules,
)

logger = logging.getLogger()


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int
    per_rank_gen: bool
    use_per_rank_bias: bool
    use_bias: bool
    per_layer_processing: bool
    use_token_mixing: bool
    num_pre_head_layers: int
    dropout_rate: float

    lora_config: LoraConfig
    extra_modules: list[str] | None
    base_hidden_size: int

    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig


def get_hypernet_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
    ctx_encoder_args: CtxEncoderArguments,
):
    num_modules = 0
    lora_config = getattr(model, "peft_config", None)
    if lora_config is not None:
        lora_config = lora_config["default"]
        num_modules += len(lora_config.target_modules)
    num_extra_modules = len(hypernet_args.extra_modules or [])
    indices = torch.arange(get_num_layers(model), device=model.device)
    return HypernetConfig(
        **vars(hypernet_args),
        base_hidden_size=model.config.hidden_size,
        lora_config=lora_config,
        layer_indices=indices,
        feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
        aggregator_config=get_aggregator_config(
            model,
            ctx_encoder_model_config,
            ctx_encoder_args.ctx_encoder_type == CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS,
            hypernet_args.latent_size,
            num_modules,
            num_extra_modules,
            lora_config.r,
            hypernet_args.per_rank_gen,
            aggregator_args,
        ),
    )


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)

    return peft_weights


class ResMLPBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        dropout_rate: float = 0,
    ):
        super().__init__()
        layers = []
        layers = [
            nn.LayerNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            nn.LayerNorm(output_size),
        ]
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class ResMLPBlockPerLayer(nn.Module):
    def __init__(
        self,
        n_layers: int,
        input_size: int,
        hidden_size: int,
        output_size: int,
    ):
        super().__init__()
        layers = [
            nn.LayerNorm(input_size),
            Mix(
                "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_hid",
                weight_shape="n_layers d_in d_hid",
                bias_shape="n_layers d_hid",
                n_layers=n_layers,
                d_in=input_size,
                d_hid=hidden_size,
            ),
            nn.SiLU(),
            Mix(
                "bs n_layers n_modules r d_hid -> bs n_layers n_modules r d_out",
                weight_shape="n_layers d_hid d_out",
                bias_shape="n_layers d_out",
                n_layers=n_layers,
                d_hid=hidden_size,
                d_out=output_size,
            ),
            nn.LayerNorm(output_size),
        ]
        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.layers(x)


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = config
        logger.debug(f"HyperLoRA config: {self.config}")
        self.iterative_mode = False
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AGGREGATOR_CLS[self.agg_config.aggregator_type](
            **vars(self.agg_config)
        )

        self.lora_config = self.config.lora_config
        self.r = self.lora_config.r

        self.target_modules = (
            tuple(sorted(self.lora_config.target_modules)) if self.lora_config else None
        )
        self.num_modules = len(self.target_modules) if self.target_modules else 0
        self.extra_modules = (
            self.config.extra_modules if self.config.extra_modules else None
        )
        self.num_extra_modules = len(self.extra_modules) if self.extra_modules else 0
        self.layer_indices = self.config.layer_indices
        self.n_layers = len(self.layer_indices)

        self.d_in, self.d_out = self.config.feature_sizes
        self.d_latent = self.config.latent_size

        if self.target_modules:
            if self.config.per_layer_processing:
                layers = [
                    ResMLPBlockPerLayer(
                        self.n_layers,
                        self.d_latent,
                        self.d_latent * 4,
                        self.d_latent,
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]
            else:
                layers = [
                    ResMLPBlock(
                        input_size=self.config.latent_size,
                        hidden_size=self.config.latent_size * 4,
                        output_size=self.config.latent_size,
                        dropout_rate=getattr(self.config, "dropout_rate", 0),
                    )
                    for _ in range(self.config.num_pre_head_layers)
                ]

            self.layers = nn.Sequential(*layers)

            self.d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

            self.bias_A = nn.ParameterDict(
                {
                    m: nn.Parameter(
                        torch.normal(
                            0,
                            0.2 / (self.d_in[m] * self.r) ** 0.5,
                            (self.n_layers, self.r, self.d_in[m]),
                        )
                    )
                    for m in self.target_modules
                }
            )
            self.bias_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((self.n_layers, self.r, self.d_out[m])))
                    for m in self.target_modules
                }
            )

            self.scaler_A = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.ones((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )
            self.scaler_B = nn.ParameterDict(
                {
                    m: nn.Parameter(torch.zeros((1, self.n_layers, self.r, 1)))
                    for m in self.target_modules
                }
            )

            n_modules = len(self.target_modules)
            # have to do this otherwise doesnt work with adamw_torch_fused
            # has something to do with the bias shape (n_modules r d_lora)
            # when n_modules == 1, adamw_torch_fused complains about device/layout
            # but when n_modules > 1, it works fine
            if n_modules == 1:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )
            else:
                self.head = Mix(
                    "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                    weight_shape="n_layers n_modules d_latent d_lora",
                    bias_shape=None,  # no bias
                    n_layers=len(self.layer_indices),
                    n_modules=n_modules,
                    d_latent=self.config.latent_size,
                    r=self.config.lora_config.r,
                    d_lora=self.d_lora,
                )

    def get_head_bias(self):
        bias_dict = dict()
        for module in self.target_modules:
            bias_A = self.bias_A[module]
            bias_B = self.bias_B[module]

            bias_dict[module] = dict(A=bias_A, B=bias_B)
        return bias_dict

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        if self.target_modules is None:
            return None
        # list of [bs, n_layers, r, in_d_outim]
        # and in_d_outim might vary across modules
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        # dict of {module:
        #   {A: [bs, n_layers, r, d_inim],
        #    B: [bs, n_layers, r, d_outim]}}
        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.d_in[module] + self.d_out[module]],
                [[self.d_in[module]], [self.d_out[module]]],
                "bs n_layers r *",
            )

            # apparently doing A * self.scaler_A is slow due to broadcasting
            A = torch.einsum("ijkl,ijkl->ijkl", A, self.scaler_A[module])
            B = torch.einsum("ijkl,ijkl->ijkl", B, self.scaler_B[module])

            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    def _to_layernorm_dict(
        self, flat_layernorms: Float[Tensor, "bs n_layers n_modules hidden_size"]
    ) -> dict[str, Float[Tensor, "bs n_layers hidden_size"]]:
        if self.extra_modules is None:
            return None
        layernorms = unpack(
            flat_layernorms,
            [[] for _ in range(len(self.extra_modules))],
            "bs n_layers * hidden_size",
        )
        return {k: v for k, v in zip(self.extra_modules, layernorms)}

    def enable_iterative_mode(self, x: bool):
        self.iterative_mode = x
        self.aggregator.enable_iterative_mode(x)

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
    ):
        # [bs, n_layers, n_total_modules, r, feature_dim]
        with torch.autocast(device_type="cuda", dtype=torch.bfloat16):
            if self.aggregator.layer_to_layer and self.iterative_mode:
                # iterative inference
                # features: [bs num_layers seq_len feature_dim]
                bs, n_layers = features.shape[0:2]
                lora_emb = torch.empty(
                    (bs, n_layers, self.num_modules, self.r, self.config.latent_size),
                    device=features.device,
                )
                for i in range(n_layers):
                    lora_emb[:, i], _ = self.aggregator(
                        features[:, i], attn_mask, position_ids
                    )

            else:
                # batched inference
                lora_emb, _ = self.aggregator(features, attn_mask, position_ids)

        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            norm = torch.norm(lora_emb, dim=-1, keepdim=True)
            norm_lora_emb = lora_emb / norm
            flat_loras = self.head(norm_lora_emb)

        flat_layernorms = None

        return flat_loras, flat_layernorms

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
        return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PeftModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        use_base_input_as_ctx: bool = False,
        # need non-packed inputs for generation
        use_sequence_packing: bool = True,
        user_defined_scaling: float = 1,
        inp_compressor=None,
    ):
        assert not use_base_input_as_ctx
        super().__init__()
        self.device = base_model.device
        self.peft_config = base_model.peft_config["default"]
        self.hypernet_config = hypernet_config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_base_input_as_ctx = use_base_input_as_ctx
        self.use_sequence_packing = use_sequence_packing
        self.user_defined_scaling = user_defined_scaling
        self.inp_compressor = inp_compressor
        self.model_accepts_loss_kwargs = True
        self.generated_loras = None

        self.register_module("base_model", base_model)
        self._init_model()
        self._bias_hyper_init()

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        train: bool = True,
        base_model_kwargs: dict = None,
        use_flash_attn: bool = True,
        **kwargs: Any,
    ):
        lora_config = state_dict["hypernet_config"].lora_config
        print(f"lora_config: {lora_config}")
        model_name_or_path = state_dict["base_model_name_or_path"]
        base_model = get_model(
            model_name_or_path,
            train=train,
            requires_grad=False,
            peft_config=lora_config,
            model_kwargs=base_model_kwargs,
            use_flash_attn=use_flash_attn,
        )
        hypernet_config = state_dict["hypernet_config"]
        if getattr(hypernet_config, "num_pre_head_layers", None) is None:
            hypernet_config.num_pre_head_layers = 4
        if getattr(hypernet_config, "use_per_rank_bias", None) is None:
            hypernet_config.use_per_rank_bias = False
        if getattr(hypernet_config, "use_bias", None) is None:
            hypernet_config.use_bias = True
        ctx_encoder_args = state_dict["ctx_encoder_args"]
        model = cls(base_model, hypernet_config, ctx_encoder_args, **kwargs)
        model.load_state_dict(state_dict)
        return model

    def patch_lora_forward(self):
        layers = get_layers(self.base_model)

        lora_forward_fn = (
            lora_forward_packed if self.use_sequence_packing else lora_forward
        )
        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                if getattr(module, "patched_forward", False):
                    continue
                logger.debug(f"Applying LoRA forward to {name}")
                module.forward_orig = module.forward
                module.patched_forward = True
                module.forward = partial(
                    lora_forward_fn,
                    self=module,
                    lora_dropout_p=self.peft_config.lora_dropout,
                    scaling=self.peft_config.lora_alpha,
                )

    def _init_model(self):
        # disable adapter of the base model
        # this only works with LoRA(?)
        # we disable to avoid peft lora computation
        self.base_model.disable_adapter_layers()

        self.hypernet = (
            HyperLoRA(self.hypernet_config).to(self.device).to(torch.float32)
        )

        self.patch_lora_forward()

        ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = self.base_model.config.name_or_path
        # use an explicit copy of the base model
        # for using with "modules_to_save"
        base_model_attn_impl = self.base_model.config._attn_implementation
        logger.debug(f"ctx_model_name: {ctx_model_name}")
        logger.debug(f"base_model.config._attn_implementation: {base_model_attn_impl}")
        encoder_model = get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=base_model_attn_impl == "flash_attention_2",
            use_q_lora=self.ctx_encoder_args.quantize_ctx_encoder,
        )
        self.ctx_encoder = CTX_ENCODER_CLS[self.ctx_encoder_args.ctx_encoder_type](
            encoder_model, self.ctx_encoder_args
        )

    # delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    @property
    def generation_config(self):
        return self.base_model.generation_config

    @property
    def vocab_size(self):
        return self.base_model.vocab_size

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @torch.no_grad()
    def _bias_hyper_init(self):
        if self.hypernet.extra_modules:
            self.hypernet.extra_head.weight.data[:] = 0
            self.hypernet.extra_head.bias.data[:] = 0
        if self.hypernet.target_modules:
            peft_weights = get_init_peft_weights(
                self.base_model, self.hypernet.lora_config
            )
            logger.debug(f"peft_weights: {peft_weights}")
            r = self.hypernet_config.lora_config.r
            nn.init.normal_(
                self.hypernet.head.weight,
                mean=0,
                std=0.5
                / sqrt(self.hypernet.config.latent_size + self.hypernet.d_lora * r),
                # the head outputs per rank lora --> divide by r to scale down grad
            )

    def state_dict(self, *args, **kwargs):
        # we assume ctx_encoder and base model is frozen here
        if len([p for p in self.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in self.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")

        state_dict = self.hypernet.state_dict(*args, **kwargs)
        state_dict["base_model_name_or_path"] = self.base_model.name_or_path
        state_dict["hypernet_config"] = self.hypernet_config
        state_dict["ctx_encoder_args"] = self.ctx_encoder_args
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        self.base_model_name_or_path = state_dict.pop("base_model_name_or_path")
        self.hypernet_config = state_dict.pop("hypernet_config")
        self.ctx_encoder_args = state_dict.pop("ctx_encoder_args")
        if self.base_model_name_or_path != self.base_model.name_or_path:
            raise ValueError(
                f"Base model name or path mismatch. "
                f"The base model given is: {self.base_model.name_or_path}, "
                f"but the loaded name is: {self.base_model_name_or_path}"
            )
        self._init_model()

        def remove_compile_prefix(sd: dict[str, Tensor]) -> dict[str, Tensor]:
            COMPILED_PREFIX = "_orig_mod."
            for k in list(sd.keys()):
                if k.startswith(COMPILED_PREFIX):
                    sd[k[len(COMPILED_PREFIX) :]] = sd.pop(k)
            return sd

        load_result = self.hypernet.load_state_dict(
            remove_compile_prefix(state_dict),
            strict=True,  # , *args, **kwargs
        )
        logger.info(f"load result: {load_result}")
        return load_result

    def generate_weights(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs ctx_len"] | None = None,
        **kwargs: Any,
    ):
        with torch.no_grad():
            ctx_encoder_kwargs = dict(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
            if isinstance(self.ctx_encoder.base_model, ModernBertModel):
                position_ids = ctx_position_ids.flatten()
                indices = torch.arange(
                    position_ids.size(0), device=position_ids.device, dtype=torch.int32
                )
                # [bsz + 1]
                cu_seqlens = torch.cat(
                    (
                        indices[position_ids == 0],
                        torch.tensor(
                            position_ids.size(),
                            device=position_ids.device,
                            dtype=torch.int32,
                        ),
                    )
                )
                ctx_encoder_kwargs = dict(
                    input_ids=ctx_ids.squeeze(0),
                    cu_seqlens=cu_seqlens,
                    max_seqlen=position_ids.max() + 1,
                    attention_mask=-1,
                    seq_len=-1,
                    batch_size=-1,
                )

            ctx_features = self.ctx_encoder(**ctx_encoder_kwargs, **kwargs)

        if isinstance(self.ctx_encoder.base_model, ModernBertModel):
            ctx_features = ctx_features.unsqueeze(0)
        if self.user_defined_scaling == 1:
            return self.hypernet.generate_weights(
                ctx_features, ctx_attn_mask, ctx_position_ids
            )

        lora_dict, _ = self.hypernet.generate_weights(
            ctx_features, ctx_attn_mask, ctx_position_ids
        )
        for module in lora_dict:
            lora_dict[module]["A"] = lora_dict[module]["A"] * self.user_defined_scaling
            lora_dict[module]["B"] = lora_dict[module]["B"] * self.user_defined_scaling
        return lora_dict, None

    def enable_iterative_mode(self, x: bool):
        self.hypernet.enable_iterative_mode(x)

    def forward(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        return_generated_lora: bool | None = False,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> tuple | ModelOutput:
        """Forward pass of the modulated model."""
        generated_loras = None
        generated_layernorms = None
        if ctx_ids is None and not self.use_base_input_as_ctx:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_features provided, using the base model for forward pass\n\n",
                    "*" * 100,
                )
            )

        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        if generated_loras is not None:
            generated_loras = combine_lora(
                generated_loras,
                n_ctx_chunks,
                lora_bias=self.hypernet.get_head_bias()
                if self.hypernet.config.use_bias
                else None,
            )

            # input_ids in model_inputs_kwargs contains only
            # prompt + response (for hypernet training)
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )

            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        ctx_ids.shape[0], dtype=torch.int32, device=self.device
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_lora_to_layers(
                self.base_model,
                self.hypernet.layer_indices,
                generated_loras,
                n_queries,
                position_ids,
            )
        model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)

        if return_generated_lora:
            return model_outputs, (generated_loras, generated_layernorms)
        else:
            return model_outputs

    def combine_lora(self, *args, **kwargs):
        # for timing
        return combine_lora(*args, **kwargs)

    def apply_lora_to_layers(self, *args, **kwargs):
        # for timing
        return apply_lora_to_layers(*args, **kwargs)

    # for simple api usage
    def internalize(self, ctx_str: str):
        ctx_tokenizer = get_tokenizer(self.ctx_encoder.base_model.name_or_path)
        ctx_ids = tokenize_ctx_text(dict(context=[ctx_str]), ctx_tokenizer)["ctx_ids"]
        return self._internalize_from_ids(torch.tensor(ctx_ids, device=self.device))

    def _internalize_from_ids(
        self,
        ctx_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_ctx ctx_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_ctx ctx_len"] | None = None,
    ):
        self.patch_lora_forward()
        if ctx_attn_mask is None and ctx_position_ids is None:
            assert ctx_ids.shape[0] == 1
            ctx_attn_mask = torch.ones_like(ctx_ids)
        generated_loras, generated_layernorms = self.generate_weights(
            ctx_ids, ctx_attn_mask, ctx_position_ids
        )
        self.generated_loras = generated_loras

    def reset(self):
        self.generated_loras = None
        layers = get_layers(self.base_model)
        for layer_idx in self.hypernet.layer_indices:
            for module_info in get_peft_modules(layers[layer_idx], self.peft_config):
                name = module_info["name"]
                module = module_info["module"]
                logger.debug(f"Resetting forward for {name}")
                module.forward = module.forward_orig
                module.patched_forward = False

    @torch.inference_mode()
    def generate(
        self,
        ctx_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_attn_mask: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        ctx_position_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        n_queries: Integer[Tensor, "n_ctx"] | None = None,
        scalers: Float[Tensor, "n_ctx"] | None = None,
        bias_scaler: float | None = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        generated_layernorms = None
        if (
            ctx_ids is None
            and not self.generated_loras
            and not self.use_base_input_as_ctx
        ):
            print(
                "*" * 100
                + "\n\nNo ctx_ids provided, using the base model for generation\n\n"
                + "*" * 100
            )
        elif ctx_ids is None and self.generated_loras:
            generated_loras = self.generated_loras
            if n_ctx_chunks is None:
                n_ctx_chunks = torch.tensor((1,), device=self.device)
            print(
                "*" * 100
                + "\n\nUsing internalized LoRAs for generation\n\n"
                + "*" * 100
            )
        else:
            if self.use_base_input_as_ctx:
                ctx_ids = (
                    model_inputs_kwargs["input_ids"]
                    if "input_ids" in model_inputs_kwargs
                    else model_inputs_args[0]
                )
                ctx_attn_mask = (
                    model_inputs_kwargs["attention_mask"]
                    if "attention_mask" in model_inputs_kwargs
                    else None
                )
                ctx_position_ids = (
                    model_inputs_kwargs["position_ids"]
                    if "position_ids" in model_inputs_kwargs
                    else None
                )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        if generated_loras is not None:
            generated_loras = self.combine_lora(
                generated_loras,
                n_ctx_chunks,
                lora_bias=self.hypernet.get_head_bias()
                if self.hypernet.config.use_bias
                else None,
                scalers=scalers,
                bias_scaler=bias_scaler,
            )

            # apply lora hook to the base model
            # TODO: we dont this position_ids for generation?
            position_ids = (
                model_inputs_kwargs["position_ids"]
                if "position_ids" in model_inputs_kwargs
                else None
            )
            if n_queries is None:
                if ctx_position_ids is None:
                    n_queries = torch.ones(
                        model_inputs_kwargs["input_ids"].shape[0],
                        dtype=torch.int32,
                        device=self.device,
                    )
                else:
                    # quite redundant (we do cu_seqlens many places)
                    # TODO: compute cu_seqlens here and propagate that
                    n_queries = torch.ones(
                        (ctx_position_ids == 0).sum(),
                        dtype=torch.int32,
                        device=self.device,
                    )

            apply_lora_to_layers(
                self.base_model,
                self.hypernet.layer_indices,
                generated_loras,
                n_queries,
                position_ids,
            )

        model_outputs = self.base_model.generate(
            *model_inputs_args, **model_inputs_kwargs
        )
        return model_outputs


# needed for loading model from checkpoint
# see https://github.com/huggingface/transformers/pull/34632
torch.serialization.add_safe_globals(
    [
        AggregatorConfig,
        LoraConfig,
        HypernetConfig,
        PeftType,
        TaskType,
        LoraRuntimeConfig,
        set,  # for real?
    ]
)
