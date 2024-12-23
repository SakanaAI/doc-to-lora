from copy import copy
from functools import partial
from importlib.resources import read_binary
import logging
import os
import random
import string
import time

import numpy as np
import torch
import yaml
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
from utils import (
    extract_cli_args,
    get_run_name,
    log_num_train_params,
    save_yaml,
    setup_logging,
    validate_args,
    validate_columns,
)

from configs import (
    ArgumentParser,
    CtxTrainingArguments,
    ExperimentSetup,
    LoRAArguments,
    ModelArguments,
)

logger = logging.getLogger()


def compute_per_token_acc(shifted_logits, shifted_labels):
    indices = np.where(shifted_labels != -100)
    acc = (shifted_logits.argmax(-1) == shifted_labels)[indices].mean()
    return {"per_token_acc": acc, "num_valid_tokens": indices[0].size}


def compute_prefix_matching(shifted_logits, shifted_labels):
    masks = np.where(shifted_labels != -100, 1, 0)
    lengths = np.sum(masks, axis=1)

    is_wrong = (shifted_logits.argmax(-1) != shifted_labels) * masks
    is_correct = (shifted_logits.argmax(-1) == shifted_labels) * masks
    # NOTE: not reliable for multi-turn conversations
    # ie, all tokens in the following user's turn will be correct
    # still monotonically correlate with perf though
    wrong_pos = np.argmax(is_wrong, axis=1) - np.argmax(masks, axis=1)
    perf = wrong_pos / lengths

    # if all tokens are correct, set to 1
    perf = np.where(is_correct.sum(axis=1) == lengths, 1, perf)
    return {"prefix_matching": perf.mean()}


def compute_entropy(shifted_logits, shifted_labels):
    indices = np.where(shifted_labels != -100)
    logits = shifted_logits[indices]
    probs = torch.softmax(torch.tensor(logits), dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return {"entropy": entropy.mean()}


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
    shifted_logits = logits[..., :-1, :]
    shifted_labels = labels[..., 1:]

    per_token_acc = compute_per_token_acc(shifted_logits, shifted_labels)
    prefix_matching = compute_prefix_matching(shifted_logits, shifted_labels)
    entropy = compute_entropy(shifted_logits, shifted_labels)
    return dict(**per_token_acc, **prefix_matching, **entropy)


def main(output_dir: str):
    ############ Argument parsing
    parser = ArgumentParser(
        (CtxTrainingArguments, ModelArguments, LoRAArguments, TrainingArguments)
    )
    ctx_args, model_args, lora_args, training_args = parser.parse()

    # there shouldn't be overlap between args
    validate_args([ctx_args, model_args, lora_args, training_args])

    args = {
        **vars(ctx_args),
        **vars(model_args),
        **vars(lora_args),
        **vars(training_args),
    }

    run_name = os.path.basename(output_dir)
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
    logger.info(f"run_name: {run_name}")
    logger.info(f"ctx_args: {ctx_args}")
    logger.info(f"model_args: {model_args}")
    logger.info(f"lora_args: {lora_args}")
    logger.debug(f"args: {args}")

    ############ Model setup

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
        # HACK: hardcode the embedding layer for now
        # TODO: add explicit encoder
        ctx_encoder = torch.nn.Embedding.from_pretrained(
            model.get_input_embeddings().weight.clone(),
            freeze=True,
        )
        model = (
            ModulatedPretrainedModel(model, hypernet, ctx_encoder)
            .to(model.device)
            .train()
        )
    else:
        # activate LoRA
        logger.info("Using LoRA")
        model.set_adapter("default")

    logger.debug(model)
    log_num_train_params(model)

    ############ Dataset setup

    logger.info("Loading dataset...")
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
    tokenized_ds.set_format(type="pt")

    validate_columns(tokenized_ds)

    train_ds = tokenized_ds["train"]
    eval_ds = {
        "train": tokenized_ds["train"].select(range(100)),
        "val": tokenized_ds["eval"],
    }

    logger.debug(f"train_ds: {train_ds}")
    logger.debug(f"eval_ds: {eval_ds}")

    # TODO: change to a faster collator? e.g.,
    # https://huggingface.co/blog/packing-with-FA2
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    def collator(inp_list, tokenizer):
        # input is a list of tokenized sequences
        padding_kwargs = dict(padding=True, pad_to_multiple_of=8, return_tensors="pt")
        labels = [x.pop("labels") for x in inp_list]
        ctx_features = None
        if "ctx_features" in inp_list[0]:
            # have to be manual since it has [ctx_len, features] shape
            ctx_features = [example.pop("ctx_features") for example in inp_list]
            ctx_features = torch.nn.utils.rnn.pad_sequence(
                ctx_features,
                batch_first=True,
                padding_value=0,
            )
            # exotic keys won't be padded, so we need to pad them as well
            ctx_attn_mask = [example.pop("ctx_attn_mask") for example in inp_list]
            ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
                ctx_attn_mask,
                batch_first=True,
                padding_value=0,
            )

        padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

        # hacky explicit padding since the labels are not padded by default
        labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
        labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
        out = {**padded_seq, "labels": labels}
        if ctx_features is not None:
            out["ctx_features"] = ctx_features
            out["ctx_attn_mask"] = ctx_attn_mask
        return out

    # TODO: use SFTTrainer instead? https://huggingface.co/docs/trl/en/sft_trainer
    # TODO: use packing with SFTTrainer

    # HACK: see transformers/trainer.py for liger-kernel patch
    # slows down training speed w/ short inputs
    # might improve/decrease training speed w/ longer inputs
    # TODO: add wandb notes somewhere
    # wandb.init(project="ctx_to_lora", name=run_name, notes=args.notes)
    train_model(
        model,
        train_ds,
        eval_ds,
        training_args,
        partial(collator, tokenizer=tokenizer),
        compute_metrics,
    )


if __name__ == "__main__":
    run_name = get_run_name()
    output_dir = f"train_outputs/{run_name}"
    setup_logging(output_dir, debug=os.environ.get("DEBUG", False))
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    save_yaml(extract_cli_args(os.sys.argv), f"{output_dir}/config.yaml")
    main(output_dir)
