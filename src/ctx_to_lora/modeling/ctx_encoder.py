import logging
from contextlib import contextmanager
from enum import Enum

import torch
from torch import nn
from transformers import PreTrainedModel

from ctx_to_lora.configs import CtxEncoderArguments
from ctx_to_lora.utils import get_base_model

logger = logging.getLogger()


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
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        base_model = get_base_model(base_model)
        if "gte" in base_model.config.name_or_path:
            base_model.encoder.layer = base_model.encoder.layer[: config.layer_idx]
        else:
            base_model.layers = base_model.layers[: config.layer_idx]

        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        model_outputs = self.base_model(**kwargs)
        return model_outputs.last_hidden_state


class EmbeddingOnly(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return the embeddings only
        return outputs.hidden_states[0]  # The first hidden state is the embeddings


class PerLayerActivations(nn.Module):
    def __init__(self, base_model: PreTrainedModel, config: CtxEncoderArguments):
        super().__init__()
        self.keep_lm_head = getattr(config, "keep_lm_head", False)
        if not self.keep_lm_head:
            base_model = get_base_model(base_model)  # remove lm head
        else:
            base_model.lm_head = nn.Identity()

        # -1 to remove last attn block
        if config.ctx_encoder_last_layer is not None:
            last_layer = config.ctx_encoder_last_layer - 1
        else:
            last_layer = -1

        if self.keep_lm_head:
            base_model.model.layers = base_model.model.layers[:last_layer]
        else:
            base_model.layers = base_model.layers[:last_layer]
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def set_input_embeddings(self, value):
        self.base_model.set_input_embeddings(value)

    def get_output_embeddings(self):
        return self.base_model.get_output_embeddings()

    def set_output_embeddings(self, new_embeddings):
        self.base_model.set_output_embeddings(new_embeddings)

    def set_decoder(self, decoder):
        self.base_model.set_decoder(decoder)

    def get_decoder(self):
        return self.base_model.get_decoder()

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return all layers' activations except the last one
        # from embeddings to the input of the last attn block
        # Shape: (batch_size, num_layers, seq_len, hidden_size)

        if self.keep_lm_head:
            return outputs
        else:
            return torch.stack(outputs.hidden_states, dim=1)


class CTX_ENCODER_TYPE(str, Enum):
    EARLY_EXIT = "early_exit"
    EMBED_ONLY = "embed_only"
    PER_LAYER_ACTIVATIONS = "per_layer_activations"


CTX_ENCODER_CLS = {
    CTX_ENCODER_TYPE.EARLY_EXIT: EarlyExit,
    CTX_ENCODER_TYPE.EMBED_ONLY: EmbeddingOnly,
    CTX_ENCODER_TYPE.PER_LAYER_ACTIVATIONS: PerLayerActivations,
}
