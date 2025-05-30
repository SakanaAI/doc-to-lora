import dataclasses
import os
import sys
from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Literal, NewType

import torch
import yaml
from transformers import (
    MODEL_FOR_CAUSAL_LM_MAPPING,
    HfArgumentParser,
    TrainingArguments,
)

MODEL_CONFIG_CLASSES = list(MODEL_FOR_CAUSAL_LM_MAPPING.keys())
MODEL_TYPES = tuple(conf.model_type for conf in MODEL_CONFIG_CLASSES)


DataClassType = NewType("DataClassType", Any)


class ArgumentParser(HfArgumentParser):
    def parse_yaml_and_args(
        self, yaml_arg: str, other_args: list[str] | None = None
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
                    if val in ["None", "none", "null", "NULL"]:
                        val = None
                        inputs[arg] = val
                        used_args[arg] = val
                        continue
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
        for arg in other_args:
            if arg not in used_args:
                raise ValueError(f"Argument provided not found in dataclass: {arg}")
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
    # LORA = "lora"
    HYPERLORA = "hyper_lora"
    # FULL_FINETUNE = "full_finetune"


@dataclass
class TrainingArguments(TrainingArguments):
    tf32: bool = field(
        default=True,
        metadata={"help": "Whether to use tf32 precision."},
    )
    dataloader_pin_memory: bool = field(
        default=True,
        metadata={"help": "Whether to pin memory in data loaders or not."},
    )
    # mem leak if use persistent workers
    # https://github.com/pytorch/pytorch/issues/62066
    # https://github.com/huggingface/transformers/issues/30943
    dataloader_persistent_workers: bool = field(
        default=False,
        metadata={
            "help": "Whether to keep the workers alive after a dataset has been consumed once."
        },
    )
    dataloader_prefetch_factor: int = field(
        default=8,
        metadata={"help": "Number of batches loaded in advance by each worker."},
    )
    dataloader_num_workers: int = field(
        default=8,
        metadata={"help": "Number of subprocesses to use for data loading."},
    )
    optim: str = field(
        default="adamw_torch_fused",
        metadata={"help": "Optimizer."},
    )
    adam_beta1: float = field(
        default=0.9,
        metadata={"help": "Adam beta 1."},
    )
    adam_beta2: float = field(
        default=0.999,
        metadata={"help": "Adam beta 2."},
    )
    adam_epsilon: float = field(
        default=1e-8,
        metadata={"help": "Adam epsilon."},
    )
    lr_scheduler_type: str = field(
        default="cosine_with_min_lr",
        metadata={"help": "Learning rate scheduler type."},
    )
    lr_scheduler_kwargs: dict = field(
        default=None,
        metadata={"help": "Learning rate scheduler kwargs."},
    )
    warmup_steps: int = field(
        default=100,
        metadata={"help": "Number of warmup steps."},
    )
    eval_on_start: bool = field(
        default=True,
        metadata={"help": "Whether to evaluate on the start of training."},
    )
    eval_strategy: str = field(
        default="steps",
        metadata={"help": "Evaluation strategy."},
    )
    eval_steps: int = field(
        default=10_000,
        metadata={"help": "Evaluation steps."},
    )
    # metric_for_best_model: str = field(
    #     default="val_loss",
    #     metadata={"help": "Metric for best model."},
    # )
    # greater_is_better: bool = field(
    #     default=False,
    #     metadata={"help": "Whether the metric is better when it is greater."},
    # )
    # load_best_model_at_end: bool = field(
    #     default=False,
    #     metadata={"help": "Whether to load the best model at the end of training."},
    # )
    save_total_limit: int = field(
        default=2,
        metadata={"help": "Total number of checkpoints to save."},
    )
    save_strategy: str = field(
        default="steps",
    )
    save_steps: int = field(
        default=10_000,
    )
    save_safetensors: bool = field(
        default=False,
    )
    logging_strategy: str = field(
        default="steps",
    )
    logging_steps: int = field(
        default=100,
    )
    use_liger_kernel: bool = field(
        default=True,
    )
    remove_unused_columns: bool = field(
        default=False,
    )
    # needed to avoid OOM by compute the metrics batch by batch
    # w/o this the trainer stores logits of all sample in memory...
    batch_eval_metrics: bool = field(
        default=True,
    )


@dataclass
class ModelArguments:
    """
    Arguments for the base model.
    """

    model_name_or_path: str = field(
        default=None,
        metadata={"help": ("Base model name or path.")},
    )
    use_flash_attn: bool = field(
        default=True,
        metadata={"help": "Whether to use flash attention."},
    )
    # use_peft: bool = field(
    #     default=False,
    #     metadata={"help": ("Whether to use PEFT or not for training.")},
    # )


@dataclass
class LoRAArguments:
    lora_r: int | None = field(
        default=8,
        metadata={"help": ("LoRA R value.")},
    )
    lora_dropout: float | None = field(
        default=0.02,
        metadata={"help": ("LoRA dropout.")},
    )
    target_modules: list[str] | None = field(
        default=None,
        metadata={"help": ("LoRA target modules.")},
    )
    # modules_to_save: Optional[list[str]] = field(
    #     default=None,
    #     metadata={"help": ("Modules to save.")},
    # )


@dataclass
class CtxTrainingArguments:
    exp_setup: ExperimentSetup = field(
        default=ExperimentSetup.HYPERLORA,
        metadata={"help": "Experiment setup - LoRA, HyperLoRA, or full finetuning"},
    )
    from_pretrained_checkpoint: str = field(
        default=None,
        metadata={"help": "Path to the pretrained checkpoint."},
    )
    max_base_len: int | None = field(
        default=2**13,
        metadata={"help": "Maximum base length for training."},
    )
    max_ctx_len: int | None = field(
        default=2**13,
        metadata={"help": "Maximum context length for training."},
    )
    use_sequence_packing: bool = field(
        default=False,
        metadata={"help": "Whether to use sequence packing."},
    )
    per_device_train_max_batch_len: int | None = field(
        default=2**12,
        metadata={
            "help": "Maximum batch length for training. Only used with multipack sampler."
        },
    )
    max_new_tokens: int | None = field(
        default=2**10,
        metadata={"help": "Maximum new tokens for generation-based evaluation."},
    )
    gen_per_device_eval_batch_size: int | None = field(
        default=1,
        metadata={"help": "Per device evaluation batch size for generation."},
    )
    notes: str | None = field(
        default=None,
        metadata={"help": "Wandb notes for the experiment."},
    )
    add_repeat_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to add repeat prompt to the dataset."},
    )
    repeat_prob: float = field(
        default=0.0,
        metadata={
            "help": "Probability that a context will be used with repeat prompts."
        },
    )
    add_negative_prompt: bool = field(
        default=False,
        metadata={"help": "Whether to add negative prompt to the dataset."},
    )
    use_kl_loss: bool = field(
        default=False,
        metadata={"help": "Whether to use KL loss."},
    )
    gen_lora_l1_reg_coef: float = field(
        default=0.0,
        metadata={"help": "L1 regularization coefficient for generated LoRAs."},
    )


@dataclass
class DataArguments:
    train_ds_names: list[str] = field(
        default=None,
        metadata={"help": "Training dataset names."},
    )
    streaming: bool = field(
        default=False,
        metadata={"help": "Whether to use streaming dataset for training."},
    )
    val_ds_names: list[str] | None = field(
        default=None,
        metadata={"help": "Validation dataset names."},
    )
    test_ds_names: list[str] | None = field(
        default=None,
        metadata={"help": "Test dataset names."},
    )
    max_val_samples_per_ds: int | None = field(
        default=5000,
        metadata={"help": "Maximum number of validation samples per dataset."},
    )
    max_test_samples_per_ds: int | None = field(
        default=1000,
        metadata={"help": "Maximum number of test samples per dataset."},
    )


@dataclass
class HypernetArguments:
    latent_size: int = field(
        default=512,
        metadata={"help": "Latent size for HyperLoRA."},
    )
    use_light_weight_lora: bool = field(
        default=False,
        metadata={"help": "Whether to use light-weight LoRA."},
    )
    light_weight_latent_size: int = field(
        default=128,
        metadata={"help": "Latent size for light-weight LoRA."},
    )
    dropout_rate: float = field(
        default=0.0,
        metadata={"help": "Dropout rate for HyperLoRA."},
    )
    extra_modules: list[str] | None = field(
        default=None,
        metadata={"help": "Extra modules to train."},
    )
    per_rank_gen: bool = field(
        default=False,
        metadata={"help": "Whether to use per-rank generation."},
    )
    per_layer_processing: bool = field(
        default=False,
        metadata={"help": "Whether to use per-layer processing (after preceiver)."},
    )
    use_token_mixing: bool = field(
        default=False,
        metadata={"help": "Whether to use token mixing block."},
    )


@dataclass
class CtxEncoderArguments:
    ctx_encoder_model_name_or_path: str = field(
        default=None,
        metadata={"help": "Context encoder model name or path."},
    )
    layer_idx: int | None = field(
        default=None,
        metadata={
            "help": "Layer index for context encoder. "
            "Default to L//4 where L is the number of layers of the ctx model"
        },
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
    # attention_probs_dropout_prob: float = field(
    #     default=0.0,
    #     metadata={"help": "Attention dropout probability for Perceiver."},
    # )
    num_latent_factor: int = field(
        default=8,
        metadata={"help": "Number of latent factors for Perceiver."},
    )
    # num_blocks: int = field(
    #     default=8,
    #     metadata={"help": "Number of blocks for Perceiver."},
    # )
    num_self_attends_per_block: int = field(
        default=6,
        metadata={"help": "Number of self-attends per block for Perceiver."},
    )
    # self_attention_widening_factor: int = field(
    #     default=4,
    #     metadata={"help": "Self-attention widening factor for Perceiver."},
    # )
    # cross_attention_widening_factor: int = field(
    #     default=4,
    #     metadata={"help": "Cross-attention widening factor for Perceiver."},
    # )
    decoder_depth: int = field(
        default=1,
        metadata={"help": "Decoder depth for Perceiver."},
    )


# needed for loading model from checkpoint
# see https://github.com/huggingface/transformers/pull/34632
torch.serialization.add_safe_globals(
    [
        DataArguments,
        CtxTrainingArguments,
        ModelArguments,
        LoRAArguments,
        TrainingArguments,
        HypernetArguments,
        AggregatorArguments,
        CtxEncoderArguments,
    ]
)


if __name__ == "__main__":
    print(ExperimentSetup)
    print(ExperimentSetup.LORA)
    print(ExperimentSetup.HYPER_LORA)
    print(ExperimentSetup.FULL_FINETUNE)
