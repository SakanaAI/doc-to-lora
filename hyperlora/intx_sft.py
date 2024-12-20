import logging
import random
import string
import time

import numpy as np
import torch
from configs import CtxTrainingArguments, ExperimentSetup, LoRAArguments, ModelArguments
from data_utils import (
    convert_ctx_prompt_response_to_messages,
    get_preprocessing_fn,
    get_sft_prompt_formatting_fn,
    tokenize_chat_messages,
)
from datasets import load_dataset

from configs import ArgumentParser
from model_loading import get_lora_config, get_model_and_tokenizer
from modeling_utils import ModulatedPretrainedModel
from training_utils import TRAINING_TASK, train_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    TrainingArguments,
)
from utils import log_num_train_params

logger = logging.getLogger(__name__)


def compute_metrics(eval_pred: EvalPrediction) -> dict:
    """
    Custom metrics function for the trainer
    Args:
        eval_pred: tuple of predictions and labels
    Returns:
        dictionary containing metric names (str) and values (Any)
    """
    # compute per token accuracy
    logits, labels = eval_pred.predictions, eval_pred.label_ids

    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    indices = np.where(shift_labels != -100)
    acc = (shift_logits.argmax(-1) == shift_labels)[indices].mean()
    return {"per_token_acc": acc, "num_valid_tokens": indices[0].size}


def main():
    # Set logging verbosity to INFO
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(
        (CtxTrainingArguments, ModelArguments, LoRAArguments, TrainingArguments)
    )
    ctx_args, model_args, lora_args, training_args = parser.parse()

    training_args.label_names = ["labels"]
    training_args.eval_on_start = True
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 500
    training_args.save_strategy = "no"
    # training_args.save_steps = 500
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 100

    uuid = "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
    )
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"

    training_args.output_dir = f"train_outputs/{run_name}"
    training_args.logging_dir = f"train_outputs/{run_name}"

    # seq2seq args for generation evaluation
    # training_args.predict_with_generate = True
    # training_args.generation_max_length = 100

    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False
    }  # manually add this argument in the code

    model_name = model_args.model_name_or_path

    model, tokenizer = get_model_and_tokenizer(
        **vars(model_args),
        train=True,
        requires_grad=ctx_args.exp_setup == ExperimentSetup.FULL_FINETUNE,
        peft_config=get_lora_config(model_name, **vars(lora_args)),
    )
    if ctx_args.exp_setup == ExperimentSetup.HYPER_LORA:
        hypernet = ...
        model = ModulatedPretrainedModel(model, hypernet)
    else:
        # activate LoRA
        model.set_adapter("default")

    print(model)
    log_num_train_params(model)

    # max_seq_len = 1024

    print("Loading dataset...")
    train_file = "data/raw_datasets/context_numbers/train.jsonl"
    eval_file = "data/raw_datasets/context_numbers/val.jsonl"
    ds = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    # preprocessing
    ds = ds.map(get_preprocessing_fn("context_numbers"))
    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)
    # for sft + chat_model, we need to convert the dataset to chat format
    # add "messages" field
    ds = ds.map(
        convert_ctx_prompt_response_to_messages,
        fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
    )
    # add "chat" field
    ds = ds.map(get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer))
    # tokenize the chat + mask the assistant inputs
    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={
            "tokenizer": tokenizer,
            "mask_assistant_inputs": True,
            "tokenizer_kwargs": {
                "max_length": None,
            },
        },
        remove_columns=ds["train"].column_names,
    )
    train_ds = tokenized_ds["train"]
    eval_ds = {
        "train": tokenized_ds["train"].select(range(100)),
        "val": tokenized_ds["eval"],
    }

    # DataCollatorForSeq2Seq also pads the `labels`
    # useful when we're computing the labels manually
    # or masking the loss only on completion
    # TODO: change to a faster collator? e.g.,
    # https://huggingface.co/blog/packing-with-FA2
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    # TODO: use SFTTrainer instead? https://huggingface.co/docs/trl/en/sft_trainer
    # TODO: use packing with SFTTrainer
    train_model(
        model,
        train_ds,
        eval_ds,
        training_args,
        data_collator,
        compute_metrics,
    )


if __name__ == "__main__":
    main()
