import dataclasses
import os
import sys
from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, NewType, Optional, Tuple
from enum import Enum, auto

from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser


class ExperimentSetup(Enum):
    LORA = "lora"
    HYPER_LORA = "hyper_lora"
    FULL_FINETUNE = "full_finetune"


@dataclass
class ModelArguments:
    """
    Arguments for the base model.
    """

    model_name_or_path: Optional[str] = field(
        default=None,
        metadata={"help": ("Base model name or path.")},
    )
    # use_peft: bool = field(
    #     default=False,
    #     metadata={"help": ("Whether to use PEFT or not for training.")},
    # )


@dataclass
class LoRAArguments:
    r: Optional[int] = field(
        default=8,
        metadata={"help": ("LoRA R value.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    target_modules: Optional[List[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )


@dataclass
class CtxTrainingArguments:
    exp_setup: ExperimentSetup = field(
        default=ExperimentSetup.LORA,
        metadata={"help": "Experiment setup - LoRA, HyperLoRA, or full finetuning"},
    )
