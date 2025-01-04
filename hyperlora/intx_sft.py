import logging
import os
import random
import string
import time
from collections import defaultdict
from copy import copy
from functools import partial
from importlib.resources import read_binary
from typing import Callable, Optional

import numpy as np
import torch
import yaml
from data_utils import (
    convert_ctx_prompt_response_to_messages,
    get_preprocessing_fn,
    get_sft_prompt_formatting_fn,
    get_tokenized_dataset,
    tokenize_chat_messages,
    tokenize_ctx_text,
)
from datasets import concatenate_datasets, disable_caching, load_dataset
from model_loading import get_lora_config, get_model_and_tokenizer
from modeling_utils import (
    EarlyExit,
    HyperLoRA,
    ModulatedPretrainedModel,
    get_hypernet_config,
)
from rouge_score import rouge_scorer
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
    get_base_model,
    get_run_name,
    log_num_train_params,
    save_yaml,
    setup_logging,
    validate_args,
)

from configs import (
    ArgumentParser,
    CtxTrainingArguments,
    DataArguments,
    ExperimentSetup,
    LoRAArguments,
    ModelArguments,
    HypernetArguments,
    AggregatorArguments,
)

logger = logging.getLogger()


def compute_per_token_acc(shift_logits, shift_labels, valid_masks):
    indices = torch.where(valid_masks)
    acc = (shift_logits.argmax(-1) == shift_labels)[indices].float().mean().item()
    return {"per_token_acc": acc}


def compute_prefix_matching(shift_logits, shift_labels, valid_masks):
    lengths = valid_masks.sum(dim=1)

    is_wrong = (shift_logits.argmax(-1) != shift_labels) * valid_masks
    is_correct = (shift_logits.argmax(-1) == shift_labels) * valid_masks
    # NOTE: not reliable for multi-turn conversations
    # ie, all tokens in the following user's turn will be correct
    # still monotonically correlate with perf though
    wrong_pos = torch.argmax(is_wrong, dim=1) - torch.argmax(valid_masks, dim=1)
    perf = wrong_pos / lengths

    # if all tokens are correct, set to 1
    perf = torch.where(is_correct.sum(dim=1) == lengths, 1, perf)
    return {"prefix_matching": perf.mean().item()}


def compute_entropy(shift_logits, shift_labels, valid_masks):
    indices = torch.where(valid_masks)
    logits = shift_logits[indices]
    probs = torch.softmax(logits, dim=-1)
    entropy = -torch.sum(probs * torch.log(probs), dim=-1)
    return {"entropy": entropy.mean().item()}


class Evaluator:
    def __init__(self, metric_fns: list[Callable]):
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        self.accum_metrics = defaultdict(list)

    def update(self, shift_logits, shift_labels, valid_masks):
        for metric_fn in self.metric_fns:
            metric = metric_fn(shift_logits, shift_labels, valid_masks)
            for k, v in metric.items():
                self.accum_metrics[k].append(v)

    def compute(self):
        # Get result across entire eval set
        result = {k: np.mean(v) for k, v in self.accum_metrics.items()}
        # Reset batch statistics
        self.reset()
        return result


def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
) -> Optional[dict]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks)
    if compute_result:
        return evaluator.compute()


# def compute_metrics(eval_pred: EvalPrediction) -> dict:
#     """
#     Custom metrics function for the trainer
#     Args:
#         eval_pred: tuple of predictions and labels
#     Returns:
#         dictionary containing metric names (str) and values (Any)
#     """
#     # compute per token accuracy
#     # logits, labels = eval_pred.predictions, eval_pred.label_ids
#     # shift_logits = logits[..., :-1, :]
#     pred_ids, labels = eval_pred.predictions, eval_pred.label_ids
#     shift_pred_ids = pred_ids[..., :-1]
#     shift_labels = labels[..., 1:]
#     valid_masks = np.where(shift_labels != -100, 1, 0)

#     per_token_acc = compute_per_token_acc(shift_pred_ids, shift_labels, valid_masks)
#     prefix_matching = compute_prefix_matching(shift_pred_ids, shift_labels, valid_masks)
#     # entropy = compute_entropy(shift_logits, shift_labels, valid_masks)
#     return dict(
#         **per_token_acc,
#         **prefix_matching,
#         # **entropy,
#         num_valid_tokens=valid_masks.sum(),
#         num_samples=valid_masks.shape[0],
#     )


# # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
# def preprocess_logits_for_metrics(logits, labels):
#     """
#     Original Trainer may have a memory leak.
#     This is a workaround to avoid storing too many tensors that are not needed.
#     """
#     pred_ids = torch.argmax(logits, dim=-1)
#     return pred_ids


def main(output_dir):
    ############ Argument parsing
    parser = ArgumentParser(
        (
            DataArguments,
            CtxTrainingArguments,
            ModelArguments,
            LoRAArguments,
            TrainingArguments,
            HypernetArguments,
            AggregatorArguments,
        )
    )
    (
        data_args,
        ctx_args,
        model_args,
        lora_args,
        training_args,
        hypernet_args,
        aggregator_args,
    ) = parser.parse()

    # there shouldn't be overlap between args
    validate_args(
        [
            data_args,
            ctx_args,
            model_args,
            lora_args,
            training_args,
            hypernet_args,
            aggregator_args,
        ]
    )

    args = {
        **vars(data_args),
        **vars(ctx_args),
        **vars(model_args),
        **vars(lora_args),
        **vars(training_args),
        **vars(hypernet_args),
        **vars(aggregator_args),
    }

    run_name = os.path.basename(output_dir)
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir
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
        hypernet = HyperLoRA(
            get_hypernet_config(model, hypernet_args, aggregator_args),
            model,
        ).to(model.device)
        # HACK: hardcode the embedding layer for now
        # TODO: add ctx_encoder config
        # ctx_encoder = torch.nn.Embedding.from_pretrained(
        #     model.get_input_embeddings().weight.clone(),
        #     freeze=True,
        # )
        # have to use at least 1 layer bc positional embeddings
        ctx_encoder = EarlyExit(get_base_model(model), 4)
        model = ModulatedPretrainedModel(model, hypernet, ctx_encoder).to(model.device)
    else:
        # activate LoRA
        logger.info("Using LoRA")
        model.set_adapter("default")

    model.train()
    logger.debug(model)
    log_num_train_params(model)

    ############ Dataset setup

    logger.info("Loading dataset...")

    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)
    need_ctx_features = isinstance(model, ModulatedPretrainedModel)
    prompt_formatting_fn = get_sft_prompt_formatting_fn(
        TRAINING_TASK.COMPLETION, tokenizer
    )
    tokenizer_kwargs = {"max_length": ctx_args.max_base_len}
    # get_ctx_features_fn = model.get_ctx_features if need_ctx_features else None
    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        add_ctx_to_chat=add_ctx_to_chat,
        # get_ctx_features_fn=get_ctx_features_fn,
    )
    tokenized_ds = {}
    for split, ds_names in zip(
        ["train", "validation", "test"],
        [data_args.train_ds_names, data_args.val_ds_names, data_args.test_ds_names],
    ):
        if ds_names is None:
            continue
        tokenized_ds[split] = concatenate_datasets(
            [_get_tokenized_dataset(ds_name, split) for ds_name in ds_names]
        )

    # for ds_name in data_args.train_ds_names:
    #     ds = load_dataset(ds_name)
    #     ds = ds.map(get_preprocessing_fn(ds_name))
    #     if "test" in ds:
    #         ds.pop("test")
    #     # check if the dataset has only ["train", "validation"]
    #     if ds.keys() > set(["train", "validation"]):
    #         raise ValueError(
    #             f"Dataset should only have 'train' and 'validation'. " f"Got {ds.keys()}"
    #         )
    #     # for sft + chat_model, we need to convert the dataset to chat format
    #     # add "messages" field
    #     ds = ds.map(
    #         convert_ctx_prompt_response_to_messages,
    #         fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
    #     )
    #     # add "chat" field
    #     # ds = ds.map(get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer))

    #     # apply chat template for chat model
    #     ds = ds.map(prompt_formatting_fn)
    #     # tokenize the chat + mask the assistant inputs
    #     pre_tok_cols = copy(ds["train"].column_names)
    #     tokenized_ds = ds.map(
    #         tokenize_chat_messages,
    #         fn_kwargs={
    #             "tokenizer": tokenizer,
    #             "mask_assistant_inputs": True,
    #             "tokenizer_kwargs": {
    #                 "max_length": ctx_args.max_base_len,
    #             },
    #         },
    #     )

    #     # computes ctx_features offline when using hyperlora
    #     if need_ctx_features:
    #         # TODO: can we batch this?
    #         # TODO: can we cache this?
    #         tokenized_ds = tokenized_ds.map(
    #             tokenize_ctx_text, fn_kwargs={"tokenizer": tokenizer}
    #         )
    #         tokenized_ds = tokenized_ds.map(
    #             # TODO: truncate the ctx_ids to the max_ctx_len
    #             model.get_ctx_features,
    #             remove_columns=["ctx_ids"],
    #         )

    #     tokenized_ds = tokenized_ds.remove_columns(pre_tok_cols)
    #     tokenized_ds.set_format(type="pt")

    #     validate_columns(tokenized_ds["train"])
    #     validate_columns(tokenized_ds["validation"])

    # TODO: add explicit validation set

    # test_ds_names = [] if data_args.test_ds_names is None else data_args.test_ds_names
    # for ds_name in test_ds_names:
    #     ds = load_dataset(ds_name)
    #     ds = ds.map(get_preprocessing_fn(ds_name))["test"]

    #     # for sft + chat_model, we need to convert the dataset to chat format
    #     # add "messages" field
    #     ds = ds.map(
    #         convert_ctx_prompt_response_to_messages,
    #         fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
    #     )
    #     # add "chat" field
    #     # ds = ds.map(get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer))

    #     # apply chat template for chat model
    #     ds = ds.map(prompt_formatting_fn)
    #     # tokenize the chat + mask the assistant inputs
    #     pre_tok_cols = copy(ds.column_names)
    #     tokenized_ds["test"] = ds.map(
    #         tokenize_chat_messages,
    #         fn_kwargs={
    #             "tokenizer": tokenizer,
    #             "mask_assistant_inputs": True,
    #             "tokenizer_kwargs": {
    #                 "max_length": ctx_args.max_base_len,
    #             },
    #         },
    #     )

    #     # computes ctx_features offline when using hyperlora
    #     if need_ctx_features:
    #         # TODO: can we batch this?
    #         # TODO: can we cache this?
    #         tokenized_ds["test"] = tokenized_ds["test"].map(
    #             tokenize_ctx_text, fn_kwargs={"tokenizer": tokenizer}
    #         )
    #         tokenized_ds["test"] = tokenized_ds["test"].map(
    #             # TODO: truncate the ctx_ids to the max_ctx_len
    #             model.get_ctx_features,
    #             remove_columns=["ctx_ids"],
    #         )

    #     tokenized_ds["test"] = tokenized_ds["test"].remove_columns(pre_tok_cols)
    #     tokenized_ds["test"].set_format(type="pt")

    #     validate_columns(tokenized_ds["test"])

    train_ds = tokenized_ds["train"]
    rand_indices = np.random.permutation(len(train_ds))[:500]
    val_ds = {
        "train": tokenized_ds["train"].select(rand_indices),
        "val": tokenized_ds.get("validation", None),
    }
    test_ds = tokenized_ds.get("test", None)

    logger.info(f"train_ds: {train_ds}")
    logger.info(f"val_ds: {val_ds}")
    logger.info(f"test_ds: {test_ds}")
    # TODO: change to a faster collator? e.g.,
    # https://huggingface.co/blog/packing-with-FA2
    # data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

    def train_collator(inp_list, tokenizer):
        # input is a list of tokenized sequences
        padding_kwargs = dict(
            padding=True,
            padding_side="right",
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        ctx_ids = None
        if "ctx_ids" in inp_list[0]:
            # have to be manual since it has [ctx_len, features] shape
            # pad to the longest ctx_len in the batch
            # which can have a different length from the input_ids, attn_mask, labels

            ctx_ids = [example.pop("ctx_ids") for example in inp_list]
            ctx_ids = torch.nn.utils.rnn.pad_sequence(
                ctx_ids,
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
        labels = [x.pop("labels") for x in inp_list]
        padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

        # hacky explicit padding since the labels are not padded by default
        labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
        labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
        out = {**padded_seq, "labels": labels}
        if ctx_ids is not None:
            out["ctx_ids"] = ctx_ids
            out["ctx_attn_mask"] = ctx_attn_mask
        return out

    def generation_collator(inp_list, tokenizer):
        padding_kwargs = dict(padding=True, padding_side="left", return_tensors="pt")
        input_ids = [x.pop("input_ids") for x in inp_list]
        attn_mask = [x.pop("attention_mask") for x in inp_list]
        labels = [x.pop("labels") for x in inp_list]
        for i, label in enumerate(labels):
            # remove the response tokens
            idx = np.argmax(label != -100)
            input_ids[i] = input_ids[i][:idx]
            attn_mask[i] = attn_mask[i][:idx]
        out = tokenizer.pad(
            {"input_ids": input_ids, "attention_mask": attn_mask}, **padding_kwargs
        )
        out["labels"] = labels

        if "ctx_ids" in inp_list[0]:
            # have to be manual since it has [ctx_len, features] shape
            # pad to the longest ctx_len in the batch
            # which can have a different length from the input_ids, attn_mask, labels
            ctx_ids = [example.pop("ctx_ids") for example in inp_list]
            ctx_ids = torch.nn.utils.rnn.pad_sequence(
                ctx_ids,
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
            out["ctx_ids"] = ctx_ids
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
        tokenizer,
        training_args,
        train_ds,
        val_ds,
        test_ds,
        partial(train_collator, tokenizer=tokenizer),
        partial(generation_collator, tokenizer=tokenizer),
        partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_entropy]
            ),
        ),
        # compute_metrics,
        # preprocess_logits_for_metrics,
    )


if __name__ == "__main__":
    run_name = get_run_name()
    output_dir = f"train_outputs/{run_name}"
    setup_logging(output_dir, debug=os.environ.get("DEBUG", False))
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    save_yaml(extract_cli_args(os.sys.argv), f"{output_dir}/config.yaml")

    # disable_caching()
    main(output_dir)
