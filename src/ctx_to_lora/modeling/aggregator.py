import logging
from dataclasses import dataclass
from enum import Enum

import torch
from einops import rearrange, repeat, unpack
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
    # n_cross_attn_layers: int  # number of cross-attention layers
    # num_self_attends_per_block: int  # = 16
    # decoder_depth: int  # 1 = only cross-attention
    num_latent_factor: int
    lora_r: int
    per_rank_gen: bool

    n_latent_queries: int  # = 208
    num_blocks: int
    num_self_attn_per_block: int
    shared_weights: bool
    layer_to_layer_ctx_encoder: bool  #  = False  # default value for backward compat


def get_aggregator_config(
    model: PreTrainedModel,
    ctx_encoder_model_config: PretrainedConfig,
    layer_to_layer_ctx_encoder: bool,
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
        layer_to_layer_ctx_encoder=layer_to_layer_ctx_encoder,
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
        num_latent_factor,  # unused
        layer_to_layer_ctx_encoder,
        n_latent_queries,
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
        self.layer_to_layer = layer_to_layer_ctx_encoder
        if self.layer_to_layer:
            n_output_queries = num_modules * self.r + num_extra_modules
        self.config = Idefics2PerceiverConfig(
            input_size=feature_size,
            # the first layer is xattn
            # resampler_depth=kwargs["num_self_attends_per_block"] + 1,
            # n_cross_attn_layers=kwargs["n_cross_attn_layers"],
            # resampler_n_latents=n_output_queries * num_latent_factor,
            num_blocks=kwargs["num_blocks"],
            num_self_attn_per_block=kwargs["num_self_attn_per_block"],
            shared_weights=kwargs["shared_weights"],
            n_latents=n_latent_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.decoder_config = Idefics2PerceiverConfig(
            input_size=output_size,
            # resampler_depth=kwargs["decoder_depth"],
            num_blocks=1,
            num_self_attn_per_block=0,
            shared_weights=False,
            n_latents=n_output_queries,
            intermediate_size_factor=4,
            hidden_size=output_size,
            attn_implementation="flash_attention_2",
        )
        self.perceiver = Idefics2Perceiver(self.config, self.decoder_config)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"]
        | Float[Tensor, "bs x seq_len feature_dim"],
        ctx_attn_mask: Integer[Tensor, "bs seq_len"] | None = None,
        ctx_position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    ):
        # if len(ctx_features.shape) == 3:
        #     # ["bs seq_len feature_dim"] to
        #     # ["bs n_output_queries feature_dim"]
        #     x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        # elif len(ctx_features.shape) == 4:
        #     # ["bs n_ctx_model_layers seq_len feature_dim"] to
        #     # to ["bs n_ctx_model_layers n_output_queries feature_dim"]
        #     x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)
        #     x = rearrange(
        #         x,
        #         (
        #             "bs n_ctx_model_layers (n_modules r) seq_len d -> "
        #             "bs (n_ctx_model_layers n_modules r) d"
        #         ),
        #         n_ctx_model_layers=self.n_ctx_model_layers,
        #         n_modules=self.num_modules,
        #         r=self.r,
        #     )
        # else:
        #     raise ValueError(
        #         f"ctx_features should be 3D or 4D tensor, got {ctx_features.shape}"
        #     )

        if self.layer_to_layer:
            if ctx_attn_mask is not None:
                ctx_attn_mask = repeat(
                    ctx_attn_mask,
                    "bs seq_len -> (num_layers bs) seq_len",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "bs num_layers seq_len feature_dim -> (num_layers bs) seq_len feature_dim",
                )
            if ctx_position_ids is not None:
                ctx_position_ids = repeat(
                    ctx_position_ids,
                    "1 seq_len -> 1 (num_layers seq_len)",
                    num_layers=self.num_layers,
                )
                ctx_features = rearrange(
                    ctx_features,
                    "1 num_layers seq_len feature_dim -> 1 (num_layers seq_len) feature_dim",
                )

        # print(f"ctx_features shape: {ctx_features.shape}")
        # print(
        #     f"ctx_attn_mask shape: {ctx_attn_mask.shape if ctx_attn_mask is not None else None}"
        # )
        # print(
        #     f"ctx_position_ids shape: {ctx_position_ids.shape if ctx_position_ids is not None else None}"
        # )
        x = self.perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)

        if self.layer_to_layer:
            per_layer_size = self.num_modules * self.r + self.num_extra_modules
            x = rearrange(
                x,
                "(num_layers bs) (per_layer_sz) d -> bs (num_layers per_layer_sz) d",
                num_layers=self.num_layers,
                per_layer_sz=per_layer_size,
            )
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


AGGREGATOR_CLS = {
    # AGGREGATOR_TYPE.POOLER: Pooler,
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}

if __name__ == "__main__":
    # Test the Perceiver class
    print("Testing Perceiver...")

    # Test parameters
    feature_size = 768
    output_size = 512
    num_layers = 12
    num_modules = 4
    num_extra_modules = 2
    lora_r = 8
    batch_size = 2
    seq_len = 100
    device = "cuda"

    torch.set_default_dtype(torch.bfloat16)

    # Create Perceiver instance
    perceiver = Perceiver(
        feature_size=feature_size,
        output_size=output_size,
        num_layers=num_layers,
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        per_rank_gen=True,
        lora_r=lora_r,
        num_latent_factor=8,
        layer_to_layer_ctx_encoder=False,
        n_latent_queries=208,
        num_self_attends_per_block=16,
        decoder_depth=1,
    ).to(device)

    # Test with 3D input (standard case)
    print("Testing with 3D input...")
    ctx_features = torch.randn(1, batch_size * seq_len, feature_size).to(device)
    ctx_attn_mask = None  # torch.ones(batch_size, seq_len, dtype=torch.long).to(device)
    ctx_position_ids = torch.arange(seq_len).repeat(batch_size).unsqueeze(0).to(device)

    with torch.no_grad():
        lora_x, extra_x = perceiver(ctx_features, ctx_attn_mask, ctx_position_ids)

    print(f"LoRA output shape: {lora_x.shape}")
    print(f"Extra output shape: {extra_x.shape}")
    print(
        f"Expected LoRA shape: ({batch_size}, {num_layers}, {num_modules}, {lora_r}, {output_size})"
    )
    print(
        f"Expected extra shape: ({batch_size}, {num_layers}, {num_extra_modules}, {output_size})"
    )

    # Test with layer-to-layer configuration
    print("\nTesting with layer-to-layer configuration...")
    perceiver_l2l = Perceiver(
        feature_size=feature_size,
        output_size=output_size,
        num_layers=num_layers,
        num_modules=num_modules,
        num_extra_modules=num_extra_modules,
        per_rank_gen=True,
        lora_r=lora_r,
        num_latent_factor=8,
        layer_to_layer_ctx_encoder=True,
        n_latent_queries=208,
        num_self_attends_per_block=16,
        decoder_depth=1,
    ).to(device)

    # Test with 4D input for layer-to-layer
    ctx_features_4d = torch.randn(1, num_layers, batch_size * seq_len, feature_size).to(
        device
    )

    with torch.no_grad():
        lora_x_l2l, extra_x_l2l = perceiver_l2l(
            ctx_features_4d, ctx_attn_mask, ctx_position_ids
        )

    print(f"Layer-to-layer LoRA output shape: {lora_x_l2l.shape}")
    print(f"Layer-to-layer extra output shape: {extra_x_l2l.shape}")

    print("All tests passed!")
