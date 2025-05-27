import logging
from contextlib import contextmanager

import torch
from torch import nn
from transformers import PreTrainedModel

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
    def __init__(self, base_model: PreTrainedModel, exit_layer: int):
        super().__init__()
        self.base_model = base_model
        if "gte" in base_model.config.name_or_path:
            self.base_model.encoder.layer = base_model.encoder.layer[:exit_layer]
        else:
            self.base_model.layers = base_model.layers[:exit_layer]

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        model_outputs = self.base_model(**kwargs)
        return model_outputs.last_hidden_state
