import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from math import log, pi
from typing import Any, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F
from einops import rearrange, repeat, unpack, einsum
from einops.layers.torch import EinMix as Mix
from einops.layers.torch import Reduce
from jaxtyping import Float, Integer
from peft import (
    get_peft_config,
    load_peft_weights,
    LoraConfig,
    PeftConfig,
    PeftModel,
    LoraRuntimeConfig,
    get_peft_model_state_dict,
    set_peft_model_state_dict,
)
from peft.utils import PeftType, TaskType, ModulesToSaveWrapper
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from torch import Tensor, nn
from transformers import (
    PerceiverConfig,
    PerceiverModel,
    PreTrainedModel,
    PretrainedConfig,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
)
from transformers.modeling_outputs import ModelOutput

from ctx_to_lora.configs import (
    AggregatorArguments,
    HypernetArguments,
    CtxEncoderArguments,
)
from ctx_to_lora.hooks import add_generated_lora_hook, remove_hook_handles
from ctx_to_lora.model_loading import get_lora_config, get_model, get_model_and_tokenizer
from ctx_to_lora.pooling import POOL_FN, get_pooling_fn
from ctx_to_lora.utils import (
    get_lora_module_names,
    get_num_layers,
    get_peft_in_out_features,
    get_base_model,
)

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE

    # pooler
    pooling_type: POOL_FN
    feature_size: int
    num_layers: int
    num_modules: int
    output_size: int

    # perceiver
    attention_probs_dropout_prob: float = 0.0
    num_blocks: int = 1
    num_self_attends_per_block: int = 16
    self_attention_widening_factor: int = 4
    cross_attention_widening_factor: int = 1
    num_latent_factor: int = 8


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    output_size: int,
    aggregator_args: AggregatorArguments,
):
    lora_config = model.peft_config["default"]
    return AggregatorConfig(
        feature_size=ctx_encoder_model_config.hidden_size,
        output_size=output_size,
        num_layers=get_num_layers(model),
        num_modules=len(lora_config.target_modules),
        **vars(aggregator_args),
    )


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int

    lora_config: LoraConfig
    module_names: dict[str, list[str]]
    # trainable_base_modules: Optional[list[str]]
    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig


def get_hypernet_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
):
    lora_config = model.peft_config["default"]
    indices = torch.arange(get_num_layers(model), device=model.device)
    return HypernetConfig(
        **vars(hypernet_args),
        lora_config=lora_config,
        module_names=get_lora_module_names(model, lora_config.target_modules, indices),
        layer_indices=indices,
        feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
        aggregator_config=get_aggregator_config(
            model,
            ctx_encoder_model_config,
            hypernet_args.latent_size,
            aggregator_args,
        ),
    )


class Perceiver(nn.Module):
    """perceiver w/ bottleneck size = n_modules * n_layers"""

    def __init__(
        self,
        feature_size,
        output_size,
        num_layers,
        num_modules,
        num_latent_factor,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.config = PerceiverConfig(
            d_model=feature_size,  # + num_bands
            num_latents=num_layers * num_modules * num_latent_factor,
            d_latents=output_size,
            # attention_probs_dropout_prob=0.0,
            # num_blocks=8,
            # num_self_attends_per_block=6,
            # self_attention_widening_factor=4,
            **kwargs,
        )
        decoder = PerceiverBasicDecoder(
            self.config,
            output_num_channels=output_size,
            output_index_dims=num_layers * num_modules,
            num_channels=output_size,
            final_project=False,
            trainable_position_encoding_kwargs=dict(
                num_channels=output_size,
                index_dims=num_layers * num_modules,
            ),
        )

        self.perceiver = PerceiverModel(self.config, decoder=decoder)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"],
        ctx_attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        x = self.perceiver(ctx_features, ctx_attn_mask).logits
        x = rearrange(
            x,
            "bs (n_layers n_modules) d -> bs n_layers n_modules d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
        )
        return x


class Mixer(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_emb_size: int,
        output_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, intermediate_emb_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_emb_size, bias=False)
        self.down_proj = nn.Linear(intermediate_emb_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pre_layer_norm: bool = True,
        post_dropout: bool = True,
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Dropout(0.05),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class Pooler(nn.Module):
    def __init__(
        self,
        feature_size: int,
        output_size: int,
        pooling_type: POOL_FN,
        num_layers: int,
        num_modules: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules

        # NOTE: features will be projected to size = output_size // 2
        # then cat with layer and module embeddings (each with size output_size // 4)
        # which are collectively form features with size = output_size
        self.pool_fn = get_pooling_fn(pooling_type)
        self.feature_proj = nn.Linear(feature_size, output_size // 2)
        self.ln = nn.LayerNorm(output_size // 2)
        self.layer_embs = nn.Sequential(
            nn.Embedding(num_layers, output_size // 4),
            nn.LayerNorm(output_size // 4),
        )
        self.module_embs = nn.Sequential(
            nn.Embedding(num_modules, output_size // 4),
            nn.LayerNorm(output_size // 4),
        )
        self.mixer = Mixer(output_size, output_size * 4, output_size)
        self.mlp = MLPResidualBlock(output_size, output_size * 4, output_size)

        self.register_buffer("layer_indices", torch.arange(num_layers))
        self.register_buffer("module_indices", torch.arange(num_modules))

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        bs = features.shape[0]

        # [bs, feature_dim]
        x = self.ln(self.feature_proj(self.pool_fn(features, attn_mask).float()))
        x = repeat(
            x,
            "bs d -> bs n_layers n_modules d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
        )

        layer_embs = self.layer_embs(self.layer_indices)  # [num_layers, d]
        layer_embs = repeat(
            layer_embs,
            "n_layers d -> bs n_layers n_modules d",
            bs=bs,
            n_modules=self.num_modules,
        )

        module_embs = self.module_embs(self.module_indices)  # [num_modules, d]
        module_embs = repeat(
            module_embs,
            "n_modules d -> bs n_layers n_modules d",
            bs=bs,
            n_layers=self.num_layers,
        )

        emb = torch.cat([x, layer_embs, module_embs], dim=3)
        return self.mlp(self.mixer(emb))


AG = {
    AGGREGATOR_TYPE.POOLER: Pooler,
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}


@contextmanager
def early_exit(base_model: PreTrainedModel, exit_layer: int):
    try:
        layers = base_model.layers
        base_model.layers = layers[:exit_layer]
        yield base_model
    finally:
        base_model.layers = layers


@contextmanager
def maybe_add_batch_dim(kwargs):
    try:
        batched_input = False
        batched_attn_mask = False
        if len(kwargs["input_ids"].shape) == 1:
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
            batched_input = True
        if len(kwargs["attention_mask"].shape) == 1:
            kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)
            batched_attn_mask = True
        yield batched_input, batched_attn_mask
    finally:
        if batched_input:
            kwargs["input_ids"] = kwargs["input_ids"].squeeze(0)
        if batched_attn_mask:
            kwargs["attention_mask"] = kwargs["attention_mask"].squeeze(0)


class EarlyExit(nn.Module):
    def __init__(self, base_model: PreTrainedModel, exit_layer: int):
        super().__init__()
        self.base_model = base_model
        self.base_model.layers = base_model.layers[:exit_layer]
        # self.exit_layer = exit_layer

    @torch.no_grad()
    def forward(self, **kwargs):
        # if len(kwargs["input_ids"].shape) == 1:
        #     kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
        # if len(kwargs["attention_mask"].shape) == 1:
        #     kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)

        with (
            # early_exit(self.base_model, self.exit_layer),
            maybe_add_batch_dim(kwargs) as (batched_input, batched_attn_mask),
        ):
            model_outputs = self.base_model(**kwargs)

        if batched_input:
            model_outputs.last_hidden_state = model_outputs.last_hidden_state.squeeze(0)

        return model_outputs.last_hidden_state


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
        assert isinstance(
            module.base_layer, nn.Linear
        ), "all modules should be a leave module that is Linear layer"

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


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = config
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AG[self.agg_config.aggregator_type](**vars(self.agg_config))

        self.lora_config = self.config.lora_config

        self.target_modules = self.lora_config.target_modules
        self.layer_indices = self.config.layer_indices

        self.d_in, self.d_out = self.config.feature_sizes

        self.layers = MLPResidualBlock(
            input_size=self.config.latent_size,
            hidden_size=self.config.latent_size * 4,
            output_size=self.config.latent_size,
        )

        if self.config.use_light_weight_lora:
            # light-weight lora projection (per module)

            self.pre_lora_projection = nn.ParameterDict(
                {
                    k: nn.Parameter(
                        torch.randn(self.d_in[k], self.config.light_weight_latent_size)
                    )
                    for k in self.target_modules
                }
            )
            for param in self.pre_lora_projection.values():
                nn.init.orthogonal_(param)

            self.post_lora_projection = nn.ParameterDict(
                {
                    k: nn.Parameter(
                        torch.randn(self.d_out[k], self.config.light_weight_latent_size)
                    )
                    for k in self.target_modules
                }
            )
            for param in self.post_lora_projection.values():
                nn.init.orthogonal_(param)

            d_lora = self.config.light_weight_latent_size * 2
            self.d_in = {k: self.config.light_weight_latent_size for k in self.d_in}
            self.d_out = {k: self.config.light_weight_latent_size for k in self.d_out}

            logger.info(f"Using light-weight LoRA with d_lora = {d_lora // 2}")
        else:
            d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

        n_modules = len(self.target_modules)
        # have to do this otherwise doesnt work with adamw_torch_fused
        # has something to do with the bias shape (n_modules r d_lora)
        # when n_modules == 1, adamw_torch_fused complains about device/layout
        # but when n_modules > 1, it works fine
        if n_modules == 1:
            self.head = Mix(
                "bs n_layers n_modules d_latent -> bs n_layers n_modules r d_lora",
                weight_shape="d_latent r d_lora",
                # bias_shape=None,  # no bias
                bias_shape="r d_lora",
                d_latent=self.config.latent_size,
                r=self.config.lora_config.r,
                d_lora=d_lora,
            )
        else:
            # each module processes d -> r d_out independently
            self.head = Mix(
                "bs n_layers n_modules d_latent -> bs n_layers n_modules r d_lora",
                weight_shape="n_modules d_latent r d_lora",
                # bias_shape=None,  # no bias
                bias_shape="n_modules r d_lora",
                n_modules=len(self.target_modules),
                d_latent=self.config.latent_size,
                r=self.config.lora_config.r,
                d_lora=d_lora,
            )
            # print(self.head)
            # print(self.head.weight.shape)
            # print(self.head.bias.shape)
            # breakpoint()

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
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
            # transpose B
            B = rearrange(B, "bs n_layers r d_out -> bs n_layers d_out r")
            if self.config.use_light_weight_lora:
                A = einsum(
                    self.pre_lora_projection[module],
                    A,
                    "d_in d_latent, bs n_layers r d_latent -> bs n_layers r d_in",
                )
                B = einsum(
                    self.post_lora_projection[module],
                    B,
                    "d_out d_latent, bs n_layers d_latent r -> bs n_layers d_out r",
                )

            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):

        # [bs, n_layers, n_modules, feature_dim]
        emb = self.aggregator(features.to(torch.float32), attn_mask)

        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = self.head(self.layers(emb))

        return flat_loras

    def generate_loras(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        flat_loras = self.forward(features, attn_mask)
        return self._to_lora_dict(flat_loras)


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        # use_kl_loss is only used for training
        use_kl_loss: bool = False,
    ):
        super().__init__()
        self.device = base_model.device
        self.hypernet_config = hypernet_config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_kl_loss = use_kl_loss

        self.register_module("base_model", base_model)
        self._init_model()
        self._bias_hyper_init()
        # self.register_module("hypernet", hypernet)
        # self.register_module("ctx_encoder", ctx_encoder)

    @classmethod
    def from_state_dict(
        cls,
        state_dict: dict,
        train: bool = True,
        use_flash_attn: bool = True,
    ):
        lora_config = state_dict["hypernet_config"].lora_config
        model_name_or_path = lora_config.base_model_name_or_path
        base_model = get_model(
            model_name_or_path,
            train=train,
            requires_grad=False,
            peft_config=lora_config,
            use_flash_attn=use_flash_attn,
        )
        hypernet_config = state_dict["hypernet_config"]
        ctx_encoder_args = state_dict["ctx_encoder_args"]
        model = cls(base_model, hypernet_config, ctx_encoder_args)
        model.load_state_dict(state_dict)
        return model

    def _init_model(self):
        # for name, module in self.base_model.named_modules():
        #     if isinstance(module, ModulesToSaveWrapper):
        #         module.set_adapter("default")
        # trainable_base_modules = getattr(
        #     self.hypernet_config, "trainable_base_modules", None
        # )
        # if trainable_base_modules:
        #     for name, param in self.base_model.named_parameters():
        #         if any(module_name in name for module_name in trainable_base_modules):
        #             param.requires_grad = True

        self.hypernet = HyperLoRA(self.hypernet_config).to(self.device)
        # ctx_encoder_name =
        # if self.ctx_encoder_args.ctx_encoder_model_name_or_path is not None:
        #     encoder_model = get_model(
        #         self.ctx_encoder_args.ctx_encoder_model_name_or_path,
        #         train=True,
        #         requires_grad=False,
        #     )
        # else:
        #     encoder_model = self.base_model
        ctx_model_name = self.ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_model_name is None:
            ctx_model_name = self.base_model.config.name_or_path
        # use an explicit copy of the base model
        # for using with "modules_to_save"
        encoder_model = get_model(
            ctx_model_name,
            train=True,
            requires_grad=False,
            use_flash_attn=self.base_model.config._attn_implementation
            == "flash_attention_2",
        )
        self.ctx_encoder = EarlyExit(
            get_base_model(encoder_model), self.ctx_encoder_args.layer_idx
        )

    def to(self, *args, **kwargs):
        # workaround to avoid the hypernet being wrapped by DeepSpeed
        self.base_model = self.base_model.to(*args, **kwargs)
        self.ctx_encoder = self.ctx_encoder.to(*args, **kwargs)
        # self.hypernet = self.hypernet.to(*args, **kwargs)
        self.hypernet.to(torch.float32)
        return self

    # Delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    @torch.no_grad()
    def _bias_hyper_init(self):
        peft_weights = get_init_peft_weights(self.base_model, self.hypernet.lora_config)
        logger.debug(f"peft_weights: {peft_weights}")
        self.hypernet.head.weight.data[:] = 0
        self.hypernet.head.bias.data[:] = 0
        # d_in = self.hypernet.d_in
        # d_out = self.hypernet.d_out
        # m = max(self.hypernet.target_modules, key=lambda m: d_in[m] + d_out[m])

        # A = peft_weights[m]["lora_A"].weight.clone()[:, : d_in[m]]  # [r, d_in]
        # B = peft_weights[m]["lora_B"].weight.clone()[: d_out[m]]  # [d_out, r]
        # biases = [A, B.T]

        # # bias-hyperinit
        # # init weights to zeros and bias to the base weights
        # bias_cat = torch.cat(biases, dim=1)
        # self.hypernet.head.bias.data[..., :, : bias_cat.shape[1]] = bias_cat
        for i, m in enumerate(self.hypernet.target_modules):
            A = peft_weights[m]["lora_A"].weight.clone()  # [r, in_d]
            B = peft_weights[m]["lora_B"].weight.clone()  # [out_d, r]
            if self.hypernet.config.use_light_weight_lora:
                A = A[:, : self.hypernet.config.light_weight_latent_size]
                B = B[: self.hypernet.config.light_weight_latent_size]
            biases = [A, B.T]

            # bias-hyperinit
            # init weights to zeros and bias to the base weights
            bias_cat = torch.cat(biases, dim=1)
            self.hypernet.head.bias.data[..., i, :, : bias_cat.shape[1]] = bias_cat
            # if len(self.hypernet.target_modules) > 1:
            #     self.hypernet.head.bias.data[..., :, : bias_cat.shape[1]] = bias_cat
            # else:
            #     self.hypernet.head.bias.data[..., :, : bias_cat.shape[1]] = bias_cat
        # self.hypernet.head.bias.requires_grad = False

    def state_dict(self, *args, **kwargs):
        # we assume ctx_encoder and base model is frozen here
        if len([p for p in self.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in self.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")

        state_dict = self.hypernet.state_dict(*args, **kwargs)
        # base_state_dict = dict()
        # if self.hypernet_config.trainable_base_modules:
        #     # for k, v in get_peft_model_state_dict(self.base_model).items():
        #     #     if any(module in k for module in self.hypernet_config.trainable_base_modules):
        #     #         # save only "modules_to_save"
        #     #         base_state_dict[k] = v
        #     for name, param in self.base_model.named_parameters():
        #         if param.requires_grad and any(
        #             module in name
        #             for module in self.hypernet_config.trainable_base_modules
        #         ):
        #             base_state_dict[name] = param.data

        # state_dict["base_model"] = base_state_dict
        state_dict["hypernet_config"] = self.hypernet_config
        state_dict["ctx_encoder_args"] = self.ctx_encoder_args
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        # NOTE: might have to set `strict=False` as we don't save all the params
        self.hypernet_config = state_dict.pop("hypernet_config")
        self.ctx_encoder_args = state_dict.pop("ctx_encoder_args")
        if (
            self.hypernet_config.lora_config.base_model_name_or_path
            != self.base_model.name_or_path
        ):
            raise ValueError(
                f"Base model name or path mismatch. "
                f"The base model given is: {self.base_model.name_or_path}, "
                f"but the hypernet config is for: {self.hypernet_config.lora_config.base_model_name_or_path}"
            )
        self._init_model()
        state_dict.pop("base_model", None)
        # base_state_dict = state_dict.pop("base_model", None)
        # if base_state_dict:
        #     print("Loading base model state dict")
        #     print(base_state_dict.keys())
        #     # peft_load_result = set_peft_model_state_dict(
        #     #     self.base_model,
        #     #     base_state_dict,
        #     #     *args,
        #     #     **kwargs,
        #     # )
        #     load_result = self.base_model.load_state_dict(base_state_dict, strict=False)
        #     print(f"base_load_result: {load_result}")
        return self.hypernet.load_state_dict(state_dict, *args, **kwargs)

    # @torch.no_grad()
    # def get_ctx_features(
    #     self,
    #     examples: dict[str, Any],
    # ):
    #     # TODO: truncate the ctx_ids to the max_ctx_len
    #     out = dict()
    #     input_ids = torch.tensor(examples.get("ctx_ids")).to(self.device)
    #     # we don't really use ctx_attn_mask here but it'll be padded
    #     # and used by hypernet.aggregator which has to handle ctx_attn_mask
    #     attention_mask = torch.tensor(examples.get("ctx_attn_mask")).to(self.device)

    #     # NOTE: this might not work for batched inputs
    #     if isinstance(self.ctx_encoder, nn.Embedding):
    #         features = self.ctx_encoder(input_ids)
    #     else:
    #         features = self.ctx_encoder(
    #             input_ids=input_ids, attention_mask=attention_mask
    #         )
    #     out["ctx_features"] = features
    #     return out

    def generate_loras(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Integer[Tensor, "bs ctx_len"],
    ):
        with torch.no_grad():
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids, attention_mask=ctx_attn_mask
            )
        return self.hypernet.generate_loras(ctx_features, ctx_attn_mask)

    def forward(
        self,
        # ctx_features: Optional[Float[Tensor, "bs ctx_length feature_dim"]] = None,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        chat_ids: Optional[Integer[Tensor, "bs chat_len"]] = None,
        chat_attn_mask: Optional[Integer[Tensor, "bs chat_len"]] = None,
        chat_labels: Optional[Integer[Tensor, "bs chat_len"]] = None,
        **model_inputs_kwargs: dict[str, Any],
    ) -> Union[tuple, ModelOutput]:
        """Forward pass of the modulated model."""

        generated_loras = None
        if ctx_ids is None:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_features provided, using the base model for forward pass\n\n",
                    "*" * 100,
                )
            )
            # model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            # return model_outputs

        else:
            with torch.no_grad():
                ctx_features = self.ctx_encoder(
                    input_ids=ctx_ids, attention_mask=ctx_attn_mask
                )
            generated_loras = self.hypernet.generate_loras(ctx_features, ctx_attn_mask)

        # compute kl loss
        # - compute logits from the base model from tokenized chat [bs, chat_len, vocab_size]
        # - compute logits from model w/ generated loras [bs, answer_len, vocab_size]
        # - select only the position of the label tokens (but we're actually not using the labels themselves)
        # - compute kl loss between the two logits
        if self.use_kl_loss and "labels" in model_inputs_kwargs:
            bs = chat_ids.shape[0]
            labels = model_inputs_kwargs.pop("labels")  # [bs, prompt_res_len]
            first_label_pos = torch.argmax((labels != -100).float(), dim=-1)
            # just a place holder, we dont use labels as groundtruth anyway
            labels[torch.arange(bs), first_label_pos - 1] = 1
            # same for chat_labels
            first_label_pos = torch.argmax((chat_labels != -100).float(), dim=-1)
            chat_labels[torch.arange(bs), first_label_pos - 1] = 1

            with torch.no_grad():
                chat_outputs = self.base_model(
                    input_ids=chat_ids, attention_mask=chat_attn_mask
                )

            with apply_generated_loras(
                self.base_model,
                generated_loras,
                self.hypernet.layer_indices,
                self.training,
            ):
                model_outputs = self.base_model(**model_inputs_kwargs)
                pred_logits = model_outputs.logits  # [bs, prompt_res_len, vocab_size]

            # [bs, ctx_prompt_res_len, vocab_size]
            base_chat_logits = chat_outputs.logits
            base_chat_logits = base_chat_logits[torch.where(chat_labels != -100)]
            pred_logits = pred_logits[torch.where(labels != -100)]
            kl_loss = F.kl_div(
                F.log_softmax(pred_logits, dim=-1),
                F.softmax(base_chat_logits, dim=-1),
                reduction="none",
            )
            # only compute kl loss on tokens where labels != -100
            kl_loss = kl_loss.sum(dim=-1).mean()
            model_outputs = ModelOutput(loss=kl_loss, **model_outputs)
        else:
            # input_ids in model_inputs_kwargs contains only
            # prompt + response (for hypernet training)
            with apply_generated_loras(
                self.base_model,
                generated_loras,
                self.hypernet.layer_indices,
                self.training,
            ):
                model_outputs = self.base_model(**model_inputs_kwargs)

        return model_outputs

    @torch.no_grad()
    def generate(
        self,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        if ctx_ids is None:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_ids provided, using the base model for generation\n\n",
                    "*" * 100,
                )
            )
            # model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            # return model_outputs
        else:
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids, attention_mask=ctx_attn_mask
            )
            generated_loras = self.hypernet.generate_loras(ctx_features, ctx_attn_mask)

        # apply lora hook to the base model
        # self.apply_generated_loras(generated_loras)
        with apply_generated_loras(
            self.base_model,
            generated_loras,
            self.hypernet.layer_indices,
            self.training,
        ):
            model_outputs = self.base_model.generate(**model_inputs_kwargs)
        return model_outputs


@contextmanager
def apply_generated_loras(
    base_model: nn.Module,
    generated_loras: Optional[dict] = None,
    layer_indices: Optional[Iterable[int]] = None,
    training: bool = False,
):
    if generated_loras is None:
        yield base_model
        return

    try:
        hooks = []
        for module_name in generated_loras:
            for layer_idx in layer_indices:
                hooks += add_generated_lora_hook(
                    base_model,
                    module_name,
                    layer_idx,
                    A=generated_loras[module_name]["A"][:, layer_idx],
                    B=generated_loras[module_name]["B"][:, layer_idx],
                    scaling=base_model.peft_config["default"].lora_alpha,
                    input_dropout=base_model.peft_config["default"].lora_dropout,
                    training=training,
                )

        yield base_model
    finally:
        remove_hook_handles(hooks)


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

if __name__ == "__main__":

    # set torch randomness seed
    torch.manual_seed(42)
    model_name = "meta-llama/Llama-3.1-8B-Instruct"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=True,
        requires_grad=False,
        peft_config=get_lora_config(model_name),
    )
    print(base_model.modules_to_save)
    ctx_model_config = base_model.config
    device = base_model.device
    # lora_config = base_model.peft_config["default"]
    # d_in, d_out = get_peft_in_out_features(base_model, peft_config=lora_config)

    hypernet_args = HypernetArguments(latent_size=512)
    aggregator_args = AggregatorArguments(aggregator_type=AGGREGATOR_TYPE.PERCEIVER)
    hypernet_config = get_hypernet_config(
        base_model, ctx_model_config, hypernet_args, aggregator_args
    )
    ctx_encoder_args = CtxEncoderArguments(layer_idx=4)
    # ctx_encoder = EarlyExit(get_base_model(base_model), 4)
    # hypernet = HyperLoRA(
    #     get_hypernet_config(base_model, hypernet_args, aggregator_args),
    #     base_model,
    # ).to(device)

    model = ModulatedPretrainedModel(base_model, hypernet_config, ctx_encoder_args).to(
        device
    )
    print(model)

    ctx_msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ctx_inputs = tokenizer(ctx_msg, return_tensors="pt").to(device)
    ctx_ids = ctx_inputs["input_ids"]
    ctx_attn_mask = ctx_inputs["attention_mask"]
    ctx_features = model.ctx_encoder(input_ids=ctx_ids, attention_mask=ctx_attn_mask).to(
        torch.float32
    )
    print(ctx_ids.shape)

    model.eval()
    agg_features = model.hypernet.aggregator(ctx_features, ctx_attn_mask)
    print(agg_features)
    print(agg_features.shape)

    hnetout = model.hypernet(ctx_features, ctx_attn_mask)
    print(hnetout)
    print(hnetout.shape)

    prompt_msg = "hello"
    prompt_inputs = tokenizer(prompt_msg, return_tensors="pt").to(device)

    basemodelout = model.base_model(**prompt_inputs)
    print(basemodelout)

    modelout = model(ctx_ids, ctx_attn_mask, **prompt_inputs)
    print(modelout)

    state_dict = torch.load(
        open(
            "train_outputs/runs/Jan15_19-10-21_slurm0-a3nodeset-11_d1842c41/checkpoint-55000/pytorch_model.bin",
            "rb",
        )
    )

    model.load_state_dict(state_dict)
    modelout = model(ctx_ids, ctx_attn_mask, **prompt_inputs)
    print(modelout)

    breakpoint()
