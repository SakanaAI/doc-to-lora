import logging
from dataclasses import dataclass, field
from enum import Enum
from functools import partial
from typing import Any, Optional, Tuple, Union

import torch
from jaxtyping import Float, Integer
from peft import LoraConfig
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from model_loading import get_lora_config, get_model_and_tokenizer

logger = logging.getLogger(__name__)


def inv_bool_mask(m: Integer[Tensor, "bs seq_len"]) -> Integer[Tensor, "bs seq_len 1"]:
    return (m - 1).bool().unsqueeze(-1)


# TODO: implement Perceiver
class Perceiver(nn.Module): ...


class MeanPool(nn.Module):
    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        if attn_mask is not None:
            features = features.masked_fill(inv_bool_mask(attn_mask), 0)
        return features.sum(dim=1, keepdim=True) / attn_mask.sum(
            dim=1, keepdim=True
        ).unsqueeze(2)


class MaxPool(nn.Module):
    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        if attn_mask is not None:
            features = features.masked_fill(inv_bool_mask(attn_mask), -float("inf"))
        return torch.max(features, dim=1, keepdim=True)


class LastTokenPool(nn.Module):
    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        last_hidden_states = (
            features["hidden_states"][-1]
            if "hidden_states" in features
            else features["last_hidden_state"]
        )

        left_padding = attn_mask[:, -1].sum() == attn_mask.shape[0]
        if left_padding:
            return last_hidden_states[:, -1]
        else:
            sequence_lengths = attn_mask.sum(dim=1) - 1
            batch_size = last_hidden_states.shape[0]
            return last_hidden_states[
                torch.arange(batch_size, device=last_hidden_states.device),
                sequence_lengths,
            ]


AGGREGATOR = Enum("AGGREGATOR", ["MEAN", "MAX", "PERCEIVER", "LAST_TOKEN"])

AGGREGATOR_CLS = {
    AGGREGATOR.MEAN: MeanPool,
    AGGREGATOR.MAX: MaxPool,
    AGGREGATOR.LAST_TOKEN: LastTokenPool,
    AGGREGATOR.PERCEIVER: Perceiver,
}


class HyperLoRA(nn.Module):
    def __init__(
        self,
        lora_config: LoraConfig,
        aggregator: AGGREGATOR,
        aggregator_kwargs: Optional[dict[str, Any]] = None,
    ):
        super().__init__()
        self.lora_config = lora_config
        self.aggregator = AGGREGATOR_CLS[aggregator](**(aggregator_kwargs or {}))

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
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=True,
        requires_grad=False,
        peft_config=get_lora_config(model_name),
    )
    print(base_model)
    hypernet = HyperLoRA(base_model.peft_config, aggregator=AGGREGATOR.MEAN)
    model = ModulatedPretrainedModel(base_model, hypernet)
    print(model)

    ctx_msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ctx_inputs = tokenizer(ctx_msg, return_tensors="pt").to(model.device)
    ctx_features = model.get_ctx_features(**ctx_inputs)
    ctx_attn_mask = ctx_inputs["attention_mask"]
    print(ctx_features.shape)

    agg_features = hypernet.aggregator(ctx_features, ctx_attn_mask)
    print(agg_features.shape)

    hnetout = hypernet(ctx_features, ctx_attn_mask)
    print(hnetout.shape)
    breakpoint()
