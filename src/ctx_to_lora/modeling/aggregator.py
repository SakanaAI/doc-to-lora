import logging
from dataclasses import dataclass
from enum import Enum

from einops import rearrange, unpack
from jaxtyping import Float, Integer
from torch import Tensor, nn
from transformers import (
    PretrainedConfig,
    PreTrainedModel,
)

from ctx_to_lora.configs import (
    AggregatorArguments,
)
from ctx_to_lora.modeling.idefics2 import Idefics2Perceiver, Idefics2PerceiverConfig
from ctx_to_lora.pooling import POOL_FN
from ctx_to_lora.utils import (
    get_num_layers,
)

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE
    num_layers: int
    num_modules: int
    num_extra_modules: int
    output_size: int
    feature_size: int

    # pooler
    pooling_type: POOL_FN

    # perceiver
    num_self_attends_per_block: int  # = 16
    decoder_depth: int  # 1 = only cross-attention
    num_latent_factor: int = 8
    lora_r: int = 8
    per_rank_gen: bool = False

    layer_to_layer_ctx_encoder: bool = False
    n_base_queries: int = 208


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
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
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


# class Pooler(nn.Module):
#     def __init__(
#         self,
#         feature_size: int,
#         output_size: int,
#         pooling_type: POOL_FN,
#         num_layers: int,
#         num_modules: int,
#         *args,
#         **kwargs,
#     ):
#         super().__init__()
#         self.num_layers = num_layers
#         self.num_modules = num_modules

#         # NOTE: features will be projected to size = output_size // 2
#         # then cat with layer and module embeddings (each with size output_size // 4)
#         # which are collectively form features with size = output_size
#         self.pool_fn = get_pooling_fn(pooling_type)
#         self.feature_proj = nn.Linear(feature_size, output_size // 2)
#         self.ln = nn.LayerNorm(output_size // 2)
#         self.layer_embs = nn.Sequential(
#             nn.Embedding(num_layers, output_size // 4),
#             nn.LayerNorm(output_size // 4),
#         )
#         self.module_embs = nn.Sequential(
#             nn.Embedding(num_modules, output_size // 4),
#             nn.LayerNorm(output_size // 4),
#         )
#         self.mixer = Mixer(output_size, output_size * 4, output_size)
#         self.mlp = MLPResidualBlock(output_size, output_size * 4, output_size)

#         self.register_buffer("layer_indices", torch.arange(num_layers))
#         self.register_buffer("module_indices", torch.arange(num_modules))

#     def forward(
#         self,
#         features: Float[Tensor, "bs seq_len feature_dim"],
#         attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
#     ):
#         bs = features.shape[0]

#         # [bs, feature_dim]
#         x = self.ln(self.feature_proj(self.pool_fn(features, attn_mask).float()))
#         x = repeat(
#             x,
#             "bs d -> bs n_layers n_modules d",
#             n_modules=self.num_modules,
#             n_layers=self.num_layers,
#         )

#         layer_embs = self.layer_embs(self.layer_indices)  # [num_layers, d]
#         layer_embs = repeat(
#             layer_embs,
#             "n_layers d -> bs n_layers n_modules d",
#             bs=bs,
#             n_modules=self.num_modules,
#         )

#         module_embs = self.module_embs(self.module_indices)  # [num_modules, d]
#         module_embs = repeat(
#             module_embs,
#             "n_modules d -> bs n_layers n_modules d",
#             bs=bs,
#             n_layers=self.num_layers,
#         )

#         emb = torch.cat([x, layer_embs, module_embs], dim=3)
#         return self.mlp(self.mixer(emb))


AG = {
    # AGGREGATOR_TYPE.POOLER: Pooler,
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}
