from copy import copy
from functools import partial
from importlib.resources import read_binary
import logging
import random
import string
import time

import numpy as np
import torch
from data_utils import (
    convert_ctx_prompt_response_to_messages,
    get_preprocessing_fn,
    get_sft_prompt_formatting_fn,
    tokenize_chat_messages,
    tokenize_ctx_text,
)
from datasets import load_dataset
from model_loading import get_lora_config, get_model_and_tokenizer
from modeling_utils import HyperLoRA, ModulatedPretrainedModel, get_hypernet_config
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

from configs import (
    ArgumentParser,
    CtxTrainingArguments,
    ExperimentSetup,
    LoRAArguments,
    ModelArguments,
)

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


def get_run_name():
    uuid = "".join(
        [random.choice(string.ascii_letters + string.digits) for _ in range(8)]
    )
    run_name = time.strftime("%Y%m%d-%H%M%S") + f"_{uuid}"
    return run_name


def main():
    # Set logging verbosity to INFO
    logging.basicConfig(level=logging.INFO)

    parser = ArgumentParser(
        (CtxTrainingArguments, ModelArguments, LoRAArguments, TrainingArguments)
    )
    ctx_args, model_args, lora_args, training_args = parser.parse()

    run_name = get_run_name()
    training_args.run_name = run_name
    training_args.output_dir = f"train_outputs/{run_name}"
    training_args.logging_dir = f"train_outputs/{run_name}"

    logger.info(f"Run name: {run_name}")
    logger.info(f"ctx_args: {ctx_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"lora_args: {lora_args}")

    model_name = model_args.model_name_or_path
    model, tokenizer = get_model_and_tokenizer(
        **vars(model_args),
        train=True,
        requires_grad=ctx_args.exp_setup == ExperimentSetup.FULL_FINETUNE,
        peft_config=get_lora_config(model_name, **vars(lora_args)),
    )

    if ctx_args.exp_setup == ExperimentSetup.HYPER_LORA:
        logger.info("Using HyperLoRA")
        hypernet = HyperLoRA(get_hypernet_config(model)).to(model.device)
        model = ModulatedPretrainedModel(model, hypernet).to(model.device).train()
    else:
        # activate LoRA
        logger.info("Using LoRA")
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
    pre_tok_cols = copy(ds["train"].column_names)
    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={
            "tokenizer": tokenizer,
            "mask_assistant_inputs": True,
            "tokenizer_kwargs": {
                "max_length": None,
            },
        },
    )

    # computes ctx_features offline when using hyperlora
    if isinstance(model, ModulatedPretrainedModel):
        # TODO: can we batch this?
        tokenized_ds = tokenized_ds.map(
            tokenize_ctx_text, fn_kwargs={"tokenizer": tokenizer}
        )
        tokenized_ds = tokenized_ds.map(
            model.get_ctx_features,
            remove_columns=["ctx_ids"],
        )

    tokenized_ds = tokenized_ds.remove_columns(pre_tok_cols)

    validate_columns(tokenized_ds)

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
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    # TODO: check tokenization pipeline (if prompt + ctx are tokenized correctly)
    def collator(inp_list, tokenizer):
        # input is a list of tokenized sequences
        padding_kwargs = dict(padding=True, pad_to_multiple_of=8, return_tensors="pt")
        labels = [x.pop("labels") for x in inp_list]
        ctx_features = None
        if "ctx_features" in inp_list[0]:
            # TODO: also pad ctx_features
            # have to be manual since it has [bs, ctx_len, features] shape
            # => pad to the max ctx_len in the batch with zeros

            # HACK: assumes ctx_features with the same size
            # only works with context_numbers
            ctx_features = torch.tensor([x.pop("ctx_features") for x in inp_list])

        padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

        # hacky explicit padding since the labels are not padded by default
        labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
        labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
        out = {**padded_seq, "labels": labels}
        if ctx_features is not None:
            out["ctx_features"] = ctx_features
        # if task_descs:
        #     task_descs = tokenizer.pad({"input_ids": task_descs}, **padding_kwargs)["input_ids"]
        #     out["task_descs_ids"] = task_descs
        return out

    # TODO: use SFTTrainer instead? https://huggingface.co/docs/trl/en/sft_trainer
    # TODO: use packing with SFTTrainer
    train_model(
        model,
        train_ds,
        eval_ds,
        training_args,
        partial(collator, tokenizer=tokenizer),
        compute_metrics,
    )


def validate_columns(tokenized_ds):
    cols = ["input_ids", "attention_mask", "labels"]
    if "ctx_features" in tokenized_ds["train"].column_names:
        cols += ["ctx_features", "ctx_attn_mask"]
    ref_cols = set(cols)
    assert (
        set(tokenized_ds["train"].column_names) == ref_cols
    ), f"Columns mismatch: {set(tokenized_ds['train'].column_names)} != {ref_cols}"


if __name__ == "__main__":
    main()
