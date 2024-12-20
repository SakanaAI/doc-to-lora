from dataclasses import dataclass, field
from typing import Any, Optional, Tuple, Union

import torch
from torch import nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

# @dataclass
# class ModelArguments:
#     model_name_or_path: str = field(default="mistralai/Mistral-7B-v0.1")
#     lora_r: int = field(default=128, metadata={"help": "lora rank"})
#     lora_dropout: float = field(default=0.05, metadata={"help": "lora dropout"})
#     train: bool = field(
#         default=True,
#         metadata={
#             "help": "if true, the model ckpt will be initialized for training; else, it's for inference"
#         },
#     )


class ModulatedPretrainedModel(nn.Module):
    # def __init__(self, model_args, training_args, lora_config):
    #     super().__init__()
    #     self.model_args = model_args
    #     self.training_args = training_args
    #     self.model_name = model_args.model_name_or_path
    #     self.lora_config = lora_config
    #     self.model = AutoModelForCausalLM.from_pretrained(
    #         self.model_name,
    #         torch_dtype=torch.float16 if training_args.bf16 is False else torch.bfloat16,
    #         use_flash_attention_2=True,
    #         resume_download=True,
    #     )
    def __init__(self, base_model: PreTrainedModel):
        super().__init__()
        self.base_model = base_model
        # NOTE: we can reduce extractor_lm's depth by
        # e.g., self.extractor_lm.model.layers = self.extractor_lm.model.layers[:num_extractor_layers]
        # or just use the token embeddings extractor_lm.embed_tokens(input_ids)
        self.extractor_lm = ...
        self.hypernet = ...

    def forward(
        self,
        ctx_ids: Optional[torch.LongTensor] = None,
        ctx_attention_mask: Optional[torch.LongTensor] = None,
        **model_inputs_kwargs: dict[str, Any],
    ) -> Union[tuple, ModelOutput]:
        """Forward pass of the modulated model.

        Args:
            ctx_ids (torch.LongTensor, optional): Token IDs to be input to the hypernet
                for generating LoRA parameters. Shape: (batch_size, ctx_length)
            input_ids (torch.LongTensor, optional): Token IDs to be input to the base
                language model. Shape: (batch_size, sequence_length)
            labels (torch.LongTensor, optional): Labels for computing the language modeling
                loss. Shape: (batch_size, sequence_length)

        Returns:
            dict: Dictionary containing:
                - loss (torch.Tensor): Language modeling loss if labels are provided
                - logits (torch.Tensor): Output logits from the model
        """
        if ctx_ids is None:
            model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            return model_outputs

        loss = ...
        logits = ...
        generated_loras = ...

        # apply lora hook to the base model
        self.apply_lora_hook(generated_loras)
        model_outputs = self.base_model(**model_inputs_kwargs)
        # model_outputs.generated_loras = generated_loras

        return model_outputs

    def apply_lora_hook(self, generated_loras):
        pass
