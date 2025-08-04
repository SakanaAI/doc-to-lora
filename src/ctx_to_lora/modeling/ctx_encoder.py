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
        base_model = get_base_model(base_model)  # remove lm head
        base_model.layers = base_model.layers[:-1]  # remove last attn block
        self.base_model = base_model

    @property
    def config(self):
        return self.base_model.config

    @torch.no_grad()
    def forward(self, **kwargs):
        kwargs["output_hidden_states"] = True  # Force output of hidden states
        outputs = self.base_model(**kwargs)
        # Return all layers' activations except the last one
        # from embeddings to the input of the last attn block
        # Shape: (batch_size, num_layers, seq_len, hidden_size)
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

if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer

    from ctx_to_lora.configs import CtxEncoderArguments

    # Test configuration
    model_name = "google/gemma-2-2b-it"
    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch.set_default_device(device)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name, device_map=device)

    # Test input
    test_text = ["Hello world", "this is a test.", "qwewq" * 50]
    inputs = tokenizer(test_text)
    position_ids = torch.cat(
        [torch.arange(len(x)) for x in inputs["input_ids"]]
    ).unsqueeze(0)
    input_ids = torch.cat([torch.tensor(x) for x in inputs["input_ids"]]).unsqueeze(0)
    inputs = {"input_ids": input_ids, "position_ids": position_ids}
    print(inputs)

    print(f"Testing all encoder classes with model: {model_name}")
    print(f"Input shape: {inputs['input_ids'].shape}")
    print("-" * 50)

    # Test EarlyExit
    # print("Testing EarlyExit...")
    config = CtxEncoderArguments(layer_idx=2)
    # early_exit_encoder = EarlyExit(base_model, config)
    # early_exit_output = early_exit_encoder(**inputs)
    # print(f"EarlyExit output shape: {early_exit_output.shape}")

    # # Test EmbeddingOnly
    # print("\nTesting EmbeddingOnly...")
    # embed_only_encoder = EmbeddingOnly(base_model, config)
    # embed_only_output = embed_only_encoder(**inputs)
    # print(f"EmbeddingOnly output shape: {embed_only_output.shape}")

    # Test PerLayerActivations
    print("\nTesting PerLayerActivations...")
    per_layer_encoder = PerLayerActivations(base_model, config)
    per_layer_output = per_layer_encoder(**inputs)
    print(f"PerLayerActivations output shape: {per_layer_output.shape}")

    print("\nAll tests completed successfully!")
