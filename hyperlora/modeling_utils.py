import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Optional, Tuple, Union
from einops import rearrange, repeat

import torch
from jaxtyping import Float, Integer
from peft import LoraConfig
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model_loading import get_lora_config, get_model_and_tokenizer
from utils import get_num_layers
from pooling import get_pooling_fn, POOL_FN

logger = logging.getLogger(__name__)


# TODO: implement Perceiver
class Perceiver(nn.Module): ...


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
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules

        self.pool_fn = get_pooling_fn(pooling_type)
        self.feature_proj = nn.Linear(feature_size, output_size)
        self.ln = nn.LayerNorm(output_size)
        self.layer_embs = nn.Sequential(
            nn.Embedding(num_layers, output_size // 2),
            nn.LayerNorm(output_size // 2),
        )
        self.module_embs = nn.Sequential(
            nn.Embedding(num_modules, output_size // 2),
            nn.LayerNorm(output_size // 2),
        )
        self.mixer = Mixer(output_size * 2, output_size * 8, output_size * 2)
        self.mlp = MLPResidualBlock(output_size * 2, output_size * 8, output_size * 2)

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
            "bs d -> bs n_modules n_layers d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
        )

        layer_embs = self.layer_embs(self.layer_indices)  # [num_layers, d]
        layer_embs = repeat(
            layer_embs,
            "n_layers d -> bs n_modules n_layers d",
            bs=bs,
            n_modules=self.num_modules,
        )

        module_embs = self.module_embs(self.module_indices)  # [num_modules, d]
        module_embs = repeat(
            module_embs,
            "n_modules d -> bs n_modules n_layers d",
            bs=bs,
            n_layers=self.num_layers,
        )

        emb = torch.cat([x, layer_embs, module_embs], dim=3)
        return self.mlp(self.mixer(emb))


AGGREGATOR = Enum("AGGREGATOR", ["POOLER", "PERCEIVER"])

AGGREGATOR_CLS = {
    # AGGREGATOR.MEAN: MeanPool,
    # AGGREGATOR.MAX: MaxPool,
    # AGGREGATOR.LAST_TOKEN: LastTokenPool,
    AGGREGATOR.POOLER: Pooler,
    AGGREGATOR.PERCEIVER: Perceiver,
}


class HyperLoRA(nn.Module):
    def __init__(
        self,
        lora_config: LoraConfig,
        layer_indices: Integer[Tensor, "num_layers"],
        aggregator: AGGREGATOR,
        aggregator_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        # TODO: aggregator should output
        # [bs, n_modules, n_layers, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers

        # NOTE: this class then only handles the output space of the hypernet
        # e.g., shared_AB_head, per_rank_gen, etc.
        self.aggregator = AGGREGATOR_CLS[aggregator](**(aggregator_kwargs or {}))

        self.lora_config = lora_config

        self.target_modules = lora_config.target_modules
        self.layer_indices = layer_indices

        self.in_features = ...
        self.out_features = ...

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ) -> Float[Tensor, "bs num_modules num_layers lora_r in_out_dim"]:

        emb = self.aggregator(features, attn_mask)  # [bs, n_features, feature_dim]
        # loras = self.layers(emb)
        # return loras
        return emb


class ModulatedPretrainedModel(nn.Module):
    def __init__(self, base_model: PreTrainedModel, hypernet: HyperLoRA):
        super().__init__()
        # self.base_model = base_model
        self.device = base_model.device
        # NOTE: we can reduce extractor_lm's depth by
        # e.g., self.extractor_lm.model.layers = self.extractor_lm.model.layers[:num_extractor_layers]
        # or just use the token embeddings extractor_lm.embed_tokens(input_ids)

        # HACK: hardcode the embedding layer for now
        # TODO: add explicit encoder
        self.encoder = base_model.get_input_embeddings()

        # register base_model as a submodule
        self.register_module("base_model", base_model)
        self.register_module("hypernet", hypernet)

    def get_ctx_features(
        self,
        input_ids: Integer[Tensor, "bs seq_len"],
        attention_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        if isinstance(self.encoder, nn.Embedding):
            features = self.encoder(input_ids)
        else:
            features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        return features

    def forward(
        self,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        **model_inputs_kwargs: dict[str, Any],
    ) -> Union[tuple, ModelOutput]:
        """Forward pass of the modulated model."""

        if ctx_ids is None:
            logger.warning(
                "No context ids provided, using the base model for the forward pass"
            )
            model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            return model_outputs

        loss = ...
        logits = ...
        # TODO: get ctx_features offline
        features = self.get_ctx_features(ctx_ids, ctx_attn_mask)
        generated_loras = self.hypernet(features, ctx_attn_mask)

        # apply lora hook to the base model
        self.apply_lora_hook(generated_loras)
        model_outputs = self.base_model(**model_inputs_kwargs)
        # model_outputs.generated_loras = generated_loras

        return model_outputs

    def apply_lora_hook(self, generated_loras):
        pass


if __name__ == "__main__":
    # set torch randomness seed
    torch.manual_seed(42)
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=True,
        requires_grad=False,
        peft_config=get_lora_config(model_name),
    )
    print(base_model)
    peft_config = base_model.peft_config["default"]

    hypernet = HyperLoRA(
        peft_config,
        torch.arange(get_num_layers(base_model), device=base_model.device),
        aggregator=AGGREGATOR.POOLER,
        aggregator_kwargs={
            "feature_size": base_model.config.hidden_size,
            "output_size": 128,
            "num_layers": get_num_layers(base_model),
            "num_modules": len(peft_config.target_modules),
            "pooling_type": POOL_FN.MEAN,
        },
    ).to(base_model.device)
    model = ModulatedPretrainedModel(base_model, hypernet).to(base_model.device)
    print(model)

    ctx_msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ctx_inputs = tokenizer(ctx_msg, return_tensors="pt").to(model.device)
    ctx_features = model.get_ctx_features(**ctx_inputs)
    ctx_attn_mask = ctx_inputs["attention_mask"]
    print(ctx_features.shape)

    hypernet.eval()
    agg_features = hypernet.aggregator(ctx_features, ctx_attn_mask)
    print(agg_features)
    print(agg_features.shape)

    hnetout = hypernet(ctx_features, ctx_attn_mask)
    print(hnetout)
    print(hnetout.shape)
    breakpoint()
