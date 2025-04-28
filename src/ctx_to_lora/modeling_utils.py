import logging
from copy import deepcopy
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
    PreTrainedTokenizerBase,
)
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
)
from transformers.modeling_outputs import ModelOutput
from transformers.models.modernbert.modeling_modernbert import ModernBertModel
from ctx_to_lora.configs import (
    AggregatorArguments,
    HypernetArguments,
    CtxEncoderArguments,
)
from ctx_to_lora.hooks import (
    add_generated_layernorm_hook,
    add_generated_lora_hook,
    remove_hook_handles,
)
from ctx_to_lora.model_loading import get_lora_config, get_model, get_model_and_tokenizer
from ctx_to_lora.pooling import POOL_FN, get_pooling_fn
from ctx_to_lora.utils import (
    get_lora_module_names,
    get_num_layers,
    get_peft_in_out_features,
    get_base_model,
)
from ctx_to_lora.modeling_idefics2 import Idefics2PerceiverConfig, Idefics2Perceiver

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
    num_extra_modules: int
    output_size: int

    # perceiver
    # attention_probs_dropout_prob: float = 0.0
    # num_blocks: int = 1
    num_self_attends_per_block: int  # = 16
    decoder_depth: int  # = 1  # 1 = only cross-attention
    # self_attention_widening_factor: int = 4
    # cross_attention_widening_factor: int = 1
    num_latent_factor: int = 8
    lora_r: int = 8
    per_rank_gen: bool = False


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    output_size: int,
    num_modules: int,
    num_extra_modules: int,
    lora_r: int,
    per_rank_gen: bool,
    aggregator_args: AggregatorArguments,
):
    return AggregatorConfig(
        feature_size=ctx_encoder_model_config.hidden_size,
        output_size=output_size,
        num_layers=get_num_layers(model),
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        lora_r=lora_r,
        per_rank_gen=per_rank_gen,
        **vars(aggregator_args),
    )


@dataclass
class HypernetConfig:
    latent_size: int
    use_light_weight_lora: bool
    light_weight_latent_size: int
    per_rank_gen: bool
    per_layer_processing: bool
    use_token_mixing: bool
    dropout_rate: float

    lora_config: LoraConfig
    # module_names: dict[str, list[str]]
    extra_modules: Optional[list[str]]
    base_hidden_size: int

    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig


def get_hypernet_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
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
        # module_names=get_lora_module_names(model, lora_config.target_modules, indices),
        layer_indices=indices,
        feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
        aggregator_config=get_aggregator_config(
            model,
            ctx_encoder_model_config,
            hypernet_args.latent_size,
            num_modules,
            num_extra_modules,
            lora_config.r,
            hypernet_args.per_rank_gen,
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
        num_extra_modules,
        per_rank_gen,
        lora_r,
        num_latent_factor,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.num_extra_modules = num_extra_modules
        self.per_rank_gen = per_rank_gen
        self.r = lora_r if self.per_rank_gen else 1
        # self.config = PerceiverConfig(
        #     d_model=feature_size,  # + num_bands
        #     num_latents=num_layers * num_modules * num_latent_factor,
        #     d_latents=output_size,
        #     # attention_probs_dropout_prob=0.0,
        #     # num_blocks=8,
        #     # num_self_attends_per_block=6,
        #     # self_attention_widening_factor=4,
        #     **kwargs,
        # )
        # decoder = PerceiverBasicDecoder(
        #     self.config,
        #     output_num_channels=output_size,
        #     output_index_dims=num_layers * num_modules,
        #     num_channels=output_size,
        #     final_project=False,
        #     trainable_position_encoding_kwargs=dict(
        #         num_channels=output_size,
        #         index_dims=num_layers * num_modules,
        #     ),
        # )

        # self.perceiver = PerceiverModel(self.config, decoder=decoder)
        n_output_queries = num_layers * (num_modules * self.r + num_extra_modules)
        self.config = Idefics2PerceiverConfig(
            input_size=feature_size,
            # the first layer is xattn
            resampler_depth=kwargs["num_self_attends_per_block"] + 1,
            resampler_n_latents=n_output_queries * num_latent_factor,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            resampler_depth=kwargs["decoder_depth"],
            resampler_n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.perceiver = Idefics2Perceiver(self.config, self.decoder_config)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"],
        ctx_attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
        ctx_position_ids: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        lora_x, extra_x = unpack(
            x,
            [
                [self.num_layers * self.num_modules * self.r],
                [self.num_layers * self.num_extra_modules],
            ],
            "bs * feature_dim",
        )
        lora_x = rearrange(
            lora_x,
            "bs (n_layers n_modules r) d -> bs n_layers n_modules r d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
            r=self.r,
        )
        if not self.per_rank_gen:
            lora_x = lora_x.squeeze(3)

        extra_x = rearrange(
            extra_x,
            "bs (n_layers n_extra_modules) d -> bs n_layers n_extra_modules d",
            n_extra_modules=self.num_extra_modules,
            n_layers=self.num_layers,
        )

        # x = rearrange(
        #     x,
        #     "bs (n_layers n_modules) d -> bs n_layers n_modules d",
        #     n_modules=self.num_modules,
        #     n_layers=self.num_layers,
        # )
        # lora_emb, extra_emb = unpack(
        #     emb,
        #     [[self.num_modules], [self.num_extra_modules]],
        #     "bs n_layers * feature_dim",
        # )
        return lora_x, extra_x


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
        self.act_fn = nn.GELU(approximate="tanh")

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
        dropout_rate: float = 0,
    ):
        super().__init__()
        layers = []
        # if pre_layer_norm:
        # layers.append(nn.LayerNorm(input_size))
        layers = [
            Gemma3RMSNorm(input_size),
            nn.Dropout(dropout_rate),
            nn.Linear(input_size, hidden_size),
            nn.GELU(approximate="tanh"),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, output_size),
            Gemma3RMSNorm(output_size),
            # nn.GELU(approximate="tanh"),
        ]
        # if post_dropout:
        #     layers.append(nn.Dropout(dropout_rate))
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
        if (
            "input_ids" in kwargs
            and kwargs["input_ids"] is not None
            and len(kwargs["input_ids"].shape) == 1
        ):
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
            batched_input = True
        if (
            "attention_mask" in kwargs
            and kwargs["attention_mask"] is not None
            and isinstance(kwargs["attention_mask"], torch.Tensor)
            and len(kwargs["attention_mask"].shape) == 1
        ):
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
        if "gte" in base_model.config.name_or_path:
            self.base_model.encoder.layer = base_model.encoder.layer[:exit_layer]
        else:
            self.base_model.layers = base_model.layers[:exit_layer]
        # self.exit_layer = exit_layer

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        # if len(kwargs["input_ids"].shape) == 1:
        #     kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
        # if len(kwargs["attention_mask"].shape) == 1:
        #     kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)

        # with (
        #     # early_exit(self.base_model, self.exit_layer),
        #     maybe_add_batch_dim(kwargs) as (batched_input, batched_attn_mask),
        # ):
        model_outputs = self.base_model(**kwargs)

        # if batched_input:
        #     model_outputs.last_hidden_state = model_outputs.last_hidden_state.squeeze(0)

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


class Gemma3RMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def _norm(self, x):
        return x * torch.rsqrt(x.pow(2).mean(-1, keepdim=True) + self.eps)

    def forward(self, x):
        output = self._norm(x.float())
        # Llama does x.to(float16) * w whilst Gemma3 is (x * w).to(float16)
        # See https://github.com/huggingface/transformers/pull/29402
        output = output * (1.0 + self.weight.float())
        return output.type_as(x)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.eps}"


class ResMLPBlock(nn.Module): ...


class UpMixPerLayer(nn.Module):
    def __init__(self, n_layers: int, n_modules: int, r: int, d_in: int, d_up: int):
        super().__init__()
        self.linear = Mix(
            "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_up",
            weight_shape="n_layers d_in d_up",
            bias_shape=None,  # "n_layers n_modules d_up",
            n_layers=n_layers,
            # n_modules=n_modules,
            r=r,
            d_in=d_in,
            d_up=d_up,
        )

    def forward(self, x):
        return self.linear(x)


class DownMixPerLayer(nn.Module):
    def __init__(self, n_layers: int, n_modules: int, r: int, d_up: int, d_out: int):
        super().__init__()
        self.linear = Mix(
            "bs n_layers n_modules r d_up -> bs n_layers n_modules r d_out",
            weight_shape="n_layers d_up d_out",
            bias_shape=None,  # "n_layers n_modules d_out",
            n_layers=n_layers,
            # n_modules=n_modules,
            r=r,
            d_up=d_up,
            d_out=d_out,
        )

    def forward(self, x):
        return self.linear(x)


class MixerPerLayer(nn.Module):
    def __init__(self, n_layers: int, n_modules: int, r: int, d_in: int):
        super().__init__()
        self.up_proj = UpMixPerLayer(n_layers, n_modules, r, d_in, d_in * 4)
        self.gate_proj = UpMixPerLayer(n_layers, n_modules, r, d_in, d_in * 4)
        self.down_proj = DownMixPerLayer(n_layers, n_modules, r, d_in * 4, d_in)
        self.act_fn = nn.GELU(approximate="tanh")

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class ResMLPBlockPerLayer(nn.Module):
    def __init__(self, n_layers: int, n_modules: int, r: int, d_in: int):
        super().__init__()
        # input shape: [bs, n_layers, n_modules, feature_dim]
        self.pre_norm = Gemma3RMSNorm(d_in)
        self.post_norm = Gemma3RMSNorm(d_in)
        self.mixer = MixerPerLayer(n_layers, n_modules, r, d_in)

    def forward(self, x):
        inp = x
        x = self.pre_norm(x)
        x = self.mixer(x)
        x = self.post_norm(x)
        return x + inp


class ResMLPTokenMixingPerLayer(nn.Module):
    def __init__(self, n_layers: int, n_modules: int, r: int, d_in: int):
        super().__init__()
        # input shape: [bs, n_layers, n_modules, feature_dim]
        d_hid = d_in // 4
        self.pre_norm = Gemma3RMSNorm(d_in)
        self.post_norm = Gemma3RMSNorm(d_in)
        # self.proj_down = Mix(
        #     "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_hid",
        #     weight_shape="n_layers d_in d_hid",
        #     bias_shape=None,  # "n_layers n_modules d_hid",
        #     n_layers=n_layers,
        #     # n_modules=n_modules,
        #     d_in=d_in,
        #     d_hid=d_hid,
        # )
        # concat all tokens (r, d_hid) then map them back to (r, d_in)
        # up_proj = Mix(
        #     "bs n_layers n_modules r d_hid -> bs n_layers n_modules r0 d_in",
        #     weight_shape="n_layers r d_hid r0 d_in",
        #     n_layers=n_layers,
        #     r=r,
        #     r0=r,
        #     d_hid=d_hid,
        #     d_in=d_in,
        # )
        # act_fn = nn.GELU(approximate="tanh")
        # out_proj = Mix(
        #     "bs n_layers n_modules r d_in -> bs n_layers n_modules r0 d_in0",
        #     weight_shape="n_layers r d_in r0 d_in0",
        #     bias_shape=None,  # "n_layers n_modules d_in0",
        #     n_layers=n_layers,
        #     r=r,
        #     r0=r,
        #     d_in=d_in,
        #     d_in0=d_in,
        # )
        # self.token_mixer = nn.Sequential(up_proj, act_fn, out_proj)
        n_token_factors = 16
        self.proj = Mix(
            "bs n_layers n_modules r d_in -> bs n_layers n_modules r d_in0",
            weight_shape="d_in d_in0",
            bias_shape=None,
            d_in=d_in,
            d_in0=d_in,
        )
        token_mix_up = Mix(
            "bs n_layers n_modules r d_in -> bs n_layers n_modules r0 d_in",
            weight_shape="n_layers r r0",
            bias_shape=None,
            n_layers=n_layers,
            r=r,
            r0=r * n_token_factors,
        )
        act_fn = nn.GELU(approximate="tanh")
        token_mix_down = Mix(
            "bs n_layers n_modules r0 d_in -> bs n_layers n_modules r d_in",
            weight_shape="n_layers r0 r",
            bias_shape=None,
            n_layers=n_layers,
            r=r,
            r0=r * n_token_factors,
        )
        self.token_mixer = nn.Sequential(token_mix_up, act_fn, token_mix_down)

    def forward(self, x):
        inp = x
        x = self.proj(x)
        x = self.pre_norm(x)
        x = self.token_mixer(x)
        x = self.post_norm(x)
        return x + inp


class HyperLoRA(nn.Module):
    def __init__(self, config: HypernetConfig):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        self.config = config
        logger.debug(f"HyperLoRA config: {self.config}")
        self._init_model()

    def _init_model(self):
        self.agg_config = self.config.aggregator_config
        self.aggregator = AG[self.agg_config.aggregator_type](**vars(self.agg_config))

        self.lora_config = self.config.lora_config

        self.target_modules = (
            self.lora_config.target_modules if self.lora_config else None
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
            # self.layers = nn.Sequential(
            #     *[
            #         MLPResidualBlock(
            #             input_size=self.config.latent_size,
            #             hidden_size=self.config.latent_size * 4,
            #             output_size=self.config.latent_size,
            #             dropout_rate=getattr(self.config, "dropout_rate", 0),
            #         )
            #         for _ in range(4)
            #     ]
            # )
            if self.config.per_layer_processing:
                layers = [
                    nn.Linear(self.d_latent, self.d_latent),
                    ResMLPBlockPerLayer(
                        self.n_layers,
                        self.num_modules,
                        self.lora_config.r,
                        self.d_latent,
                    ),
                ]
            else:
                layers = [
                    nn.Linear(self.d_latent, self.d_latent),
                    MLPResidualBlock(
                        input_size=self.config.latent_size,
                        hidden_size=self.config.latent_size * 4,
                        output_size=self.config.latent_size,
                        dropout_rate=getattr(self.config, "dropout_rate", 0),
                    ),
                ]
            if self.config.use_token_mixing:
                layers = [
                    ResMLPTokenMixingPerLayer(
                        self.n_layers,
                        self.num_modules,
                        self.lora_config.r,
                        self.d_latent,
                    )
                ] + layers

            self.layers = nn.Sequential(*layers)

            d_lora = max(self.d_in[m] + self.d_out[m] for m in self.target_modules)

            # if self.config.use_light_weight_lora:
            #     # light-weight lora projection (per module)

            #     self.pre_lora_projection = nn.ParameterDict(
            #         {
            #             k: nn.Parameter(
            #                 torch.randn(
            #                     self.d_in[k], self.config.light_weight_latent_size
            #                 )
            #             )
            #             for k in self.target_modules
            #         }
            #     )
            #     for param in self.pre_lora_projection.values():
            #         nn.init.orthogonal_(param)

            #     self.post_lora_projection = nn.ParameterDict(
            #         {
            #             k: nn.Parameter(
            #                 torch.randn(
            #                     self.d_out[k], self.config.light_weight_latent_size
            #                 )
            #             )
            #             for k in self.target_modules
            #         }
            #     )
            #     for param in self.post_lora_projection.values():
            #         nn.init.orthogonal_(param)

            #     d_lora = self.config.light_weight_latent_size * 2
            #     self.d_in = {k: self.config.light_weight_latent_size for k in self.d_in}
            #     self.d_out = {
            #         k: self.config.light_weight_latent_size for k in self.d_out
            #     }

            #     logger.info(f"Using light-weight LoRA with d_lora = {d_lora // 2}")

            n_modules = len(self.target_modules)
            # have to do this otherwise doesnt work with adamw_torch_fused
            # has something to do with the bias shape (n_modules r d_lora)
            # when n_modules == 1, adamw_torch_fused complains about device/layout
            # but when n_modules > 1, it works fine
            if self.config.per_rank_gen:
                if n_modules == 1:
                    if self.config.per_layer_processing:
                        self.head = Mix(
                            "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_layers d_latent d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_layers d_lora",
                            n_layers=len(self.layer_indices),
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
                    else:
                        self.head = Mix(
                            "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="d_latent d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="d_lora",
                            # n_layers=len(self.layer_indices),
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
                else:
                    if self.config.per_layer_processing:
                        self.head = Mix(
                            "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_layers n_modules r d_latent d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_layers n_modules r d_lora",
                            n_layers=len(self.layer_indices),
                            n_modules=n_modules,
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
                    else:
                        self.head = Mix(
                            "bs n_layers n_modules r d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_modules r d_latent d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_modules r d_lora",
                            n_layers=len(self.layer_indices),
                            n_modules=n_modules,
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
            else:
                if n_modules == 1:
                    if self.config.per_layer_processing:
                        self.head = Mix(
                            "bs n_layers n_modules d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_layers d_latent r d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_layers r d_lora",
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
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
                    if self.config.per_layer_processing:
                        self.head = Mix(
                            "bs n_layers n_modules d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_layers n_modules d_latent r d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_layers n_modules r d_lora",
                            n_modules=n_modules,
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )
                    else:
                        self.head = Mix(
                            "bs n_layers n_modules d_latent -> bs n_layers n_modules r d_lora",
                            weight_shape="n_modules d_latent r d_lora",
                            # bias_shape=None,  # no bias
                            bias_shape="n_modules r d_lora",
                            n_modules=n_modules,
                            d_latent=self.config.latent_size,
                            r=self.config.lora_config.r,
                            d_lora=d_lora,
                        )

        # if self.extra_modules:
        #     # self.extra_layers = nn.Sequential(
        #     #     *[
        #     #         MLPResidualBlock(
        #     #             input_size=self.config.latent_size,
        #     #             hidden_size=self.config.latent_size * 4,
        #     #             output_size=self.config.latent_size,
        #     #             dropout_rate=getattr(self.config, "dropout_rate", 0),
        #     #         )
        #     #         for _ in range(4)
        #     #     ]
        #     # )
        #     self.extra_layers = MLPResidualBlock(
        #         input_size=self.config.latent_size,
        #         hidden_size=self.config.latent_size * 4,
        #         output_size=self.config.latent_size,
        #         dropout_rate=getattr(self.config, "dropout_rate", 0),
        #     )
        #     # self.extra_layers = nn.Identity()
        #     # self.pre_extra_layers_norm = nn.LayerNorm(self.config.latent_size)
        #     # self.extra_norm = nn.LayerNorm(self.config.latent_size)

        #     # separate head for extra_modules, e.g., layernorm
        #     n_extra_modules = len(self.extra_modules) if self.extra_modules else 0
        #     if n_extra_modules == 1:
        #         self.extra_head = Mix(
        #             "bs n_layers n_modules d_latent -> bs n_layers n_modules hidden_size",
        #             weight_shape="d_latent hidden_size",
        #             bias_shape="hidden_size",
        #             d_latent=self.config.latent_size,
        #             hidden_size=self.config.base_hidden_size,
        #         )
        #     else:
        #         self.extra_head = Mix(
        #             "bs n_layers n_modules d_latent -> bs n_layers n_modules hidden_size",
        #             weight_shape="n_modules d_latent hidden_size",
        #             bias_shape="n_modules hidden_size",
        #             n_modules=n_extra_modules,
        #             d_latent=self.config.latent_size,
        #             hidden_size=self.config.base_hidden_size,
        #         )

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

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
        position_ids: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):

        # [bs, n_layers, n_total_modules, feature_dim]
        lora_emb, extra_emb = self.aggregator(features, attn_mask, position_ids)
        # lora_emb, extra_emb = unpack(
        #     emb,
        #     [[self.num_modules], [self.num_extra_modules]],
        #     "bs n_layers * feature_dim",
        # )

        # [bs, n_layers, n_modules, r, max_in_d_outim]
        flat_loras = None
        if self.target_modules:
            lora_emb = self.layers(lora_emb)
            flat_loras = self.head(lora_emb)

        flat_layernorms = None
        if self.num_extra_modules:
            # [bs, n_layers, n_extra_modules, base_hidden_size]
            extra_emb = self.extra_layers(extra_emb)
            flat_layernorms = self.extra_head(extra_emb)

        return flat_loras, flat_layernorms

    def generate_weights(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
        position_ids: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        flat_loras, flat_layernorms = self.forward(features, attn_mask, position_ids)
        # print(f"flat_loras: {flat_loras.shape}")
        # if flat_layernorms is not None:
        #     print(f"flat_layernorms: {flat_layernorms.shape}")
        return self._to_lora_dict(flat_loras), self._to_layernorm_dict(flat_layernorms)


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,
        hypernet_config: HypernetConfig,
        ctx_encoder_args: CtxEncoderArguments,
        # use_kl_loss is only used for training
        use_kl_loss: bool = False,
        use_base_input_as_ctx: bool = False,
    ):
        super().__init__()
        self.device = base_model.device
        self.hypernet_config = hypernet_config
        self.ctx_encoder_args = ctx_encoder_args
        self.use_kl_loss = use_kl_loss
        self.use_base_input_as_ctx = use_base_input_as_ctx

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
        model_name_or_path = state_dict["base_model_name_or_path"]
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

        self.hypernet = HyperLoRA(self.hypernet_config).to(self.device).to(torch.float32)
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
        base_model_attn_impl = self.base_model.config._attn_implementation
        logger.debug(f"ctx_model_name: {ctx_model_name}")
        logger.debug(f"base_model.config._attn_implementation: {base_model_attn_impl}")
        encoder_model = get_model(
            ctx_model_name,
            train=self.base_model.training,
            requires_grad=False,
            use_flash_attn=base_model_attn_impl == "flash_attention_2",
        )
        self.ctx_encoder = EarlyExit(
            get_base_model(encoder_model), self.ctx_encoder_args.layer_idx
        )

    # def to(self, *args, **kwargs):
    #     # workaround to avoid the hypernet being wrapped by DeepSpeed
    #     self.base_model = self.base_model.to(*args, **kwargs)
    #     self.ctx_encoder = self.ctx_encoder.to(*args, **kwargs)
    #     # self.hypernet = self.hypernet.to(*args, **kwargs)
    #     # self.hypernet.to(torch.float32)
    #     return self

    # Delegate to base_model
    @property
    def config(self):
        return self.base_model.config

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
                if self.hypernet.config.per_rank_gen:
                    A = A[0:1]
                    B = B[:, 0:1]
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
        state_dict["base_model_name_or_path"] = self.base_model.name_or_path
        state_dict["hypernet_config"] = self.hypernet_config
        state_dict["ctx_encoder_args"] = self.ctx_encoder_args
        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        # NOTE: might have to set `strict=False` as we don't save all the params
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

    def generate_weights(
        self,
        ctx_ids: Integer[Tensor, "bs ctx_len"],
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        ctx_position_ids: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        **kwargs: Any,
    ):
        with torch.no_grad():
            ctx_encoder_kwargs = dict(
                input_ids=ctx_ids,
                attention_mask=ctx_attn_mask,
                position_ids=ctx_position_ids,
            )
            # TODO: for modernbert ctx_encoder pass
            # `cu_seq_len` and `max_seq_len` to the forward call
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

        # print(f"padded ctx_features: {ctx_features.shape}")
        # if ctx_attn_mask is not None:
        #     print(f"padded ctx_attn_mask: {ctx_attn_mask.shape}")
        # if ctx_position_ids is not None:
        #     print(f"padded ctx_position_ids: {ctx_position_ids.shape}")

        return self.hypernet.generate_weights(
            ctx_features, ctx_attn_mask, ctx_position_ids
        )

    # @torch.inference_mode()
    # def _unpack(self, ctx_position_ids, ctx_features):
    #     # [1, len, d] -> [len, d]
    #     ctx_features = ctx_features.squeeze(0)
    #     lengths = torch.where(ctx_position_ids == 0)[1]
    #     total_len = torch.tensor([len(ctx_features)], device=self.device)
    #     lengths = torch.cat([lengths, total_len])
    #     # compute the difference between the lengths
    #     lens = torch.diff(lengths).unsqueeze(1)

    #     ctx_features = unpack(ctx_features, lens, "* d")
    #     ctx_attn_mask = [torch.ones(len(x), device=self.device) for x in ctx_features]
    #     ctx_features = torch.nn.utils.rnn.pad_sequence(
    #         ctx_features,
    #         batch_first=True,
    #         padding_value=0,
    #     )
    #     ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
    #         ctx_attn_mask,
    #         batch_first=True,
    #         padding_value=0,
    #     )

    #     return ctx_features, ctx_attn_mask

    def forward(
        self,
        # ctx_features: Optional[Float[Tensor, "bs ctx_length feature_dim"]] = None,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        ctx_position_ids: Optional[Integer[Tensor, "bs ctx_len"]] = None,
        return_generated_lora: Optional[bool] = False,
        # chat_ids: Optional[Integer[Tensor, "bs chat_len"]] = None,
        # chat_attn_mask: Optional[Integer[Tensor, "bs chat_len"]] = None,
        # chat_labels: Optional[Integer[Tensor, "bs chat_len"]] = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ) -> Union[tuple, ModelOutput]:
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
            # model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            # return model_outputs

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
            # with torch.inference_mode():
            #     ctx_features = self.ctx_encoder(
            #         input_ids=ctx_ids,
            #         attention_mask=ctx_attn_mask,
            #         position_ids=ctx_position_ids,
            #     )
            # generated_loras, generated_layernorms = self.hypernet.generate_weights(
            #     ctx_features, ctx_attn_mask
            # )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        # input_ids in model_inputs_kwargs contains only
        # prompt + response (for hypernet training)
        position_ids = (
            model_inputs_kwargs["position_ids"]
            if "position_ids" in model_inputs_kwargs
            else None
        )
        with (
            apply_generated_loras(
                self.base_model,
                generated_loras,
                self.hypernet.layer_indices,
                position_ids,
                self.training,
            ),
            apply_generated_layernorm(
                self.base_model,
                generated_layernorms,
                self.hypernet.layer_indices,
                position_ids,
                self.training,
            ),
        ):
            model_outputs = self.base_model(*model_inputs_args, **model_inputs_kwargs)

        if return_generated_lora:
            return model_outputs, (generated_loras, generated_layernorms)
        else:
            return model_outputs

    @torch.inference_mode()
    def generate(
        self,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_position_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        *model_inputs_args: Any,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        generated_layernorms = None
        if ctx_ids is None and not self.use_base_input_as_ctx:
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
            # ctx_features = self.ctx_encoder(
            #     input_ids=ctx_ids, attention_mask=ctx_attn_mask
            # )
            # generated_loras, generated_layernorms = self.hypernet.generate_weights(
            #     ctx_features, ctx_attn_mask
            # )
            generated_loras, generated_layernorms = self.generate_weights(
                ctx_ids, ctx_attn_mask, ctx_position_ids
            )

        # apply lora hook to the base model
        # self.apply_generated_loras(generated_loras)
        position_ids = (
            model_inputs_kwargs["position_ids"]
            if "position_ids" in model_inputs_kwargs
            else None
        )
        with (
            apply_generated_loras(
                self.base_model,
                generated_loras,
                self.hypernet.layer_indices,
                position_ids,
                self.training,
            ),
            apply_generated_layernorm(
                self.base_model,
                generated_layernorms,
                self.hypernet.layer_indices,
                position_ids,
                self.training,
            ),
        ):
            model_outputs = self.base_model.generate(
                *model_inputs_args, **model_inputs_kwargs
            )
        return model_outputs


class ModulatedModelWithSharedInput(nn.Module):
    def __init__(
        self,
        modulated_model: ModulatedPretrainedModel,
        base_tokenizer: PreTrainedTokenizerBase,
        ctx_tokenizer: PreTrainedTokenizerBase,
        ctx_end_predicate: Optional[str] = None,
        remove_ctx_from_base_input: bool = False,
    ):
        super().__init__()
        self.modulated_model = modulated_model
        self.base_tokenizer = base_tokenizer
        self.ctx_tokenizer = ctx_tokenizer
        self.register_module("modulated_model", self.modulated_model)
        self.ctx_end_predicate = ctx_end_predicate
        self.remove_ctx_from_base_input = remove_ctx_from_base_input

    # delegate to self.modulated_model
    @property
    def config(self):
        return self.modulated_model.config

    @property
    def device(self):
        return self.modulated_model.device

    def tie_weights(self):
        self.modulated_model.base_model.tie_weights()

    def state_dict(self, *args, **kwargs):
        return self.modulated_model.state_dict(*args, **kwargs)

    def forward(self, *args, **kwargs):
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        input_txts = self.base_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        ctx_txts = deepcopy(input_txts)
        if self.ctx_end_predicate:
            ctx_txts = [
                txt.split(self.ctx_end_predicate)[0].strip() for txt in input_txts
            ]
        ctx_inputs = self.ctx_tokenizer(ctx_txts, return_tensors="pt", padding=True).to(
            self.device
        )
        ctx_ids, ctx_attn_mask = ctx_inputs.input_ids, ctx_inputs.attention_mask
        if self.remove_ctx_from_base_input:
            raise NotImplementedError("Not implemented")
            # input_txts = [
            #     txt.split(self.ctx_end_predicate)[1].strip() for txt in input_txts
            # ]
            # inputs = self.base_tokenizer(input_txts, return_tensors="pt", padding=True)
            # input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            # kwargs["input_ids"] = input_ids
            # kwargs["attention_mask"] = attention_mask
        return self.modulated_model(ctx_ids, ctx_attn_mask, *args, **kwargs)

    def generate(self, *args, **kwargs):
        input_ids = kwargs["input_ids"] if "input_ids" in kwargs else args[0]
        input_txts = self.base_tokenizer.batch_decode(
            input_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        ctx_txts = deepcopy(input_txts)
        if self.ctx_end_predicate:
            ctx_txts = [
                txt.split(self.ctx_end_predicate)[0].strip() for txt in input_txts
            ]
        ctx_inputs = self.ctx_tokenizer(ctx_txts, return_tensors="pt", padding=True).to(
            self.device
        )
        ctx_ids, ctx_attn_mask = ctx_inputs.input_ids, ctx_inputs.attention_mask
        if self.remove_ctx_from_base_input:
            raise NotImplementedError("Not implemented")
            # input_txts = [
            #     txt.split(self.ctx_end_predicate)[1].strip() for txt in input_txts
            # ]
            # inputs = self.base_tokenizer(input_txts, return_tensors="pt", padding=True)
            # input_ids, attention_mask = inputs.input_ids, inputs.attention_mask
            # kwargs["input_ids"] = input_ids
            # kwargs["attention_mask"] = attention_mask
        return self.modulated_model.generate(ctx_ids, ctx_attn_mask, *args, **kwargs)


@contextmanager
def apply_generated_layernorm(
    base_model: nn.Module,
    generated_layernorms: Optional[dict[str, Float[Tensor, "bs n_layers h"]]] = None,
    layer_indices: Optional[Iterable[int]] = None,
    position_ids: Optional[Integer[Tensor, "bs seq_len"]] = None,
    training: bool = False,
):
    if generated_layernorms is None:
        yield base_model
        return

    try:
        hooks = []
        for module_name in generated_layernorms:
            for layer_idx in layer_indices:
                hooks += add_generated_layernorm_hook(
                    base_model,
                    module_name,
                    layer_idx,
                    W=generated_layernorms[module_name][:, layer_idx],
                    position_ids=position_ids,
                    training=training,
                )
        yield base_model
    finally:
        remove_hook_handles(hooks)


@contextmanager
def apply_generated_loras(
    base_model: nn.Module,
    generated_loras: Optional[dict] = None,
    layer_indices: Optional[Iterable[int]] = None,
    position_ids: Optional[Integer[Tensor, "bs seq_len"]] = None,
    training: bool = False,
):
    if generated_loras is None:
        yield base_model
        return

    try:
        hooks = []
        for module_name in generated_loras:
            for layer_idx in layer_indices:
                # TODO: handle sequence packing???
                hooks += add_generated_lora_hook(
                    base_model,
                    module_name,
                    layer_idx,
                    A=generated_loras[module_name]["A"][:, layer_idx],
                    B=generated_loras[module_name]["B"][:, layer_idx],
                    scaling=base_model.peft_config["default"].lora_alpha,
                    input_dropout=base_model.peft_config["default"].lora_dropout,
                    position_ids=position_ids,
                    training=training,
                )

        yield base_model
    finally:
        remove_hook_handles(hooks)


@contextmanager
def apply_generated_loras_packed_sequence(
    base_model: nn.Module,
    generated_loras: Optional[dict] = None,
    layer_indices: Optional[Iterable[int]] = None,
    training: bool = False,
):
    pass


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
        "train_outputs/runs/Jan15_19-10-21_slurm0-a3nodeset-11_d1842c41/checkpoint-55000/pytorch_model.bin",
    )
    print(state_dict.keys())
    breakpoint()

    model.load_state_dict(state_dict)
    modelout = model(ctx_ids, ctx_attn_mask, **prompt_inputs)
    print(modelout)

    breakpoint()
