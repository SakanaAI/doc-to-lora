import dataclasses
import os
import sys
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Any, Dict, List, Literal, NewType, Optional, Tuple

import yaml
from transformers import MODEL_FOR_CAUSAL_LM_MAPPING, HfArgumentParser

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: Optional[list[str]] = None
    ) -> list[dataclass]:
        """
        Parse a YAML file and overwrite the default/loaded values with the values provided to the command line.

        Args:
            yaml_arg (`str`):
                The path to the config file used
            other_args (`List[str]`, *optional`):
                A list of strings to parse as command line arguments, e.g. ['--arg=val', '--arg2=val2'].

        Returns:
            [`List[dataclass]`]: a list of dataclasses with the values from the YAML file and the command line
        """
        arg_list = self.parse_yaml_file(os.path.abspath(yaml_arg))

        outputs = []
        # strip other args list into dict of key-value pairs
        other_args = {
            arg.split("=")[0].strip("-"): arg.split("=")[1] for arg in other_args
        }
        used_args = {}

        # overwrite the default/loaded value with the value provided to the command line
        # adapted from https://github.com/huggingface/transformers/blob/d0b5002378daabf62769159add3e7d66d3f83c3b/src/transformers/hf_argparser.py#L327
        for data_yaml, data_class in zip(arg_list, self.dataclass_types):
            keys = {f.name for f in dataclasses.fields(data_yaml) if f.init}
            inputs = {k: v for k, v in vars(data_yaml).items() if k in keys}
            for arg, val in other_args.items():
                # add only if in keys
                if arg in keys:
                    base_type = data_yaml.__dataclass_fields__[arg].type
                    inputs[arg] = val

                    # cast type for ints, floats (default to strings)
                    if base_type in [int, float]:
                        inputs[arg] = base_type(val)

                    if base_type == list[str]:
                        inputs[arg] = [str(v) for v in val.split(",")]

                    # bool of a non-empty string is True, so we manually check for bools
                    if base_type == bool:
                        if val in ["true", "True"]:
                            inputs[arg] = True
                        else:
                            inputs[arg] = False

                    if base_type == dict:
                        inputs[arg] = yaml.load(val, Loader=yaml.FullLoader)

                    # add to used-args so we can check if double add
                    if arg not in used_args:
                        used_args[arg] = val
                    else:
                        raise ValueError(
                            f"Duplicate argument provided: {arg}, may cause unexpected behavior"
                        )
                # else:
                #     raise ValueError(f"Argument provided not found in dataclass: {arg}")

            obj = data_class(**inputs)
            outputs.append(obj)

        return outputs

    def parse(self) -> DataClassType | tuple[DataClassType]:
        if len(sys.argv) == 2 and sys.argv[1].endswith(".yaml"):
            # If we pass only one argument to the script and it's the path to a YAML file,
            # let's parse it to get our arguments.
            output = self.parse_yaml_file(os.path.abspath(sys.argv[1].split("=")[-1]))
        # parse command line args and yaml file
        elif len(sys.argv) > 2 and sys.argv[1].endswith(".yaml"):
            output = self.parse_yaml_and_args(
                os.path.abspath(sys.argv[1].split("=")[-1]), sys.argv[2:]
            )
        # parse --config for the yaml path and other command line args
        elif any([arg.startswith("--config") for arg in sys.argv]):
            yaml_arg = [
                arg
                for arg in sys.argv[1:]
                if arg.startswith("--config") and arg.endswith(".yaml")
            ][0]
            other_args = [arg for arg in sys.argv[1:] if arg != yaml_arg]
            output = self.parse_yaml_and_args(
                os.path.abspath(yaml_arg.split("=")[-1]), other_args
            )
        # parse command line args only
        else:
            output = self.parse_args_into_dataclasses()

        if len(output) == 1:
            output = output[0]
        return output


class ExperimentSetup(str, Enum):
    LORA = "lora"
    HYPER_LORA = "hyper_lora"
    FULL_FINETUNE = "full_finetune"


@dataclass
class ModelArguments:
    """
    Arguments for the base model.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": ("Base model name or path.")},
    )
    # use_peft: bool = field(
    #     default=False,
    #     metadata={"help": ("Whether to use PEFT or not for training.")},
    # )


@dataclass
class LoRAArguments:
    lora_r: Optional[int] = field(
        default=8,
        metadata={"help": ("LoRA R value.")},
    )
    lora_dropout: Optional[float] = field(
        default=0.05,
        metadata={"help": ("LoRA dropout.")},
    )
    target_modules: Optional[list[str]] = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )


@dataclass
class CtxTrainingArguments:
    exp_setup: ExperimentSetup = field(
        default=ExperimentSetup.LORA,
        metadata={"help": "Experiment setup - LoRA, HyperLoRA, or full finetuning"},
    )
    max_base_len: Optional[int] = field(
        default=2**13,
        metadata={"help": "Maximum base length for training."},
    )


@dataclass
class DataArguments:
    train_ds_names: list[str] = field(
        default=None,
        metadata={"help": "Training dataset names."},
    )
    val_ds_names: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Validation dataset names."},
    )
    test_ds_names: Optional[list[str]] = field(
        default=None,
        metadata={"help": "Test dataset names."},
    )
    max_val_samples: Optional[int] = field(
        default=5000,
        metadata={"help": "Maximum number of validation samples."},
    )


@dataclass
class HypernetArguments:
    latent_size: int = field(
        default=512,
        metadata={"help": "Latent size for HyperLoRA."},
    )


@dataclass
class CtxEncoderArguments:
    layer_idx: int = field(
        default=4,
        metadata={"help": "Layer index for context encoder."},
    )


@dataclass
class AggregatorArguments:

    aggregator_type: Literal["pooler", "perceiver"] = field(
        default="pooler",
        metadata={"help": "Aggregator type for HyperLoRA."},
    )

    # pooler
    pooling_type: str = field(
        default="mean",
        metadata={"help": "Pooling type for HyperLoRA."},
    )
    # feature_size: int
    # num_layers: int
    # num_modules: int
    # output_size: int

    # perceiver
    attention_probs_dropout_prob: float = field(
        default=0.0,
        metadata={"help": "Attention dropout probability for Perceiver."},
    )
    num_blocks: int = field(
        default=8,
        metadata={"help": "Number of blocks for Perceiver."},
    )
    num_self_attends_per_block: int = field(
        default=6,
        metadata={"help": "Number of self-attends per block for Perceiver."},
    )
    self_attention_widening_factor: int = field(
        default=1,
        metadata={"help": "Self-attention widening factor for Perceiver."},
    )
    cross_attention_widening_factor: int = field(
        default=1,
        metadata={"help": "Cross-attention widening factor for Perceiver."},
    )


if __name__ == "__main__":
    print(ExperimentSetup)
    print(ExperimentSetup.LORA)
    print(ExperimentSetup.HYPER_LORA)
    print(ExperimentSetup.FULL_FINETUNE)
