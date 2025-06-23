import itertools
from collections.abc import Iterable

import numpy as np
import torch
from transformers.data import (
    DataCollatorWithFlattening,
    default_data_collator,
)

from ctx_to_lora.utils import concat_list

flattener = DataCollatorWithFlattening()


def concat_batch(inp_list):
    sample = inp_list[0]
    out = [
        {
            k: list(itertools.chain.from_iterable([x[k] for x in inp_list]))
            for k in sample
        }
    ]
    return out


def flatten_if_not_packed(inp_list):
    # TODO: handle new data format
    # no padding
    sample = inp_list[0]
    n = len(sample)
    if "position_ids" in sample:
        if n == 1:
            return default_data_collator(inp_list, return_tensors="pt")
        elif n > 1:
            # needed for when batch_size > 1 (never used?)
            return default_data_collator(concat_batch(inp_list), return_tensors="pt")

    # for eval data during training
    packed_inputs = flattener(inp_list, return_tensors="pt")
    if "ctx_ids" in sample:
        ctx_ids = [{"input_ids": example["ctx_ids"]} for example in inp_list]
        packed_ctx = flattener(ctx_ids, return_tensors="pt")
        packed_inputs["ctx_ids"] = packed_ctx["input_ids"]
        packed_inputs["ctx_position_ids"] = packed_ctx["position_ids"]
        # for eval info
        if "ctx_ids_len" in sample:
            packed_inputs["ctx_ids_len"] = [
                example["ctx_ids_len"] for example in inp_list
            ]

    return packed_inputs


def train_collator(inp_list, tokenizer):
    # input is a list of tokenized sequences
    padding_kwargs = dict(
        padding=True,
        padding_side="right",
        pad_to_multiple_of=8,
        return_tensors="pt",
    )
    sample = inp_list[0]
    ctx_ids = None
    if "ctx_ids" in sample:
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

    need_flatten = isinstance(sample["input_ids"][0], Iterable)

    if need_flatten:
        # needed for one-lora-multi-q
        n_queries = torch.tensor([len(x["input_ids"]) for x in inp_list])
        tokens = dict()
        for k in ["input_ids", "attention_mask", "labels"]:
            tokens[k] = concat_list([x[k] for x in inp_list])

        labels = tokens.pop("labels")
        padded_seq = tokenizer.pad(tokens, **padding_kwargs)
        # hacky explicit padding since the labels are not padded by default
        labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    else:
        # needed for eval data during training
        labels = [x.pop("labels") for x in inp_list]
        padded_seq = tokenizer.pad(inp_list, **padding_kwargs)
        n_queries = torch.ones(len(inp_list), dtype=torch.int32)

    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels, "n_queries": n_queries}
    if ctx_ids is not None:
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask

    return out


def eval_collator(inp_list, tokenizer):
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

    chat_ids = None
    if "chat_ids" in inp_list[0]:
        chat_ids = [x.pop("chat_ids") for x in inp_list]
        chat_ids = torch.nn.utils.rnn.pad_sequence(
            chat_ids,
            batch_first=True,
            padding_value=0,
        )
        chat_attn_mask = [x.pop("chat_attn_mask") for x in inp_list]
        chat_attn_mask = torch.nn.utils.rnn.pad_sequence(
            chat_attn_mask,
            batch_first=True,
            padding_value=0,
        )
        chat_labels = [x.pop("chat_labels") for x in inp_list]
        chat_labels = torch.nn.utils.rnn.pad_sequence(
            chat_labels,
            batch_first=True,
            padding_value=-100,
        )
        chat_labels = torch.where(chat_attn_mask == 0, -100, chat_labels)

    labels = [x.pop("labels") for x in inp_list]
    padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

    # hacky explicit padding since the labels are not padded by default
    labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels}
    if ctx_ids is not None:
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask
    if chat_ids is not None:
        out["chat_ids"] = chat_ids
        out["chat_attn_mask"] = chat_attn_mask
        out["chat_labels"] = chat_labels
    # if "input_ids_len" in inp_list[0]:
    #     out["input_ids_len"] = [x.pop("input_ids_len") for x in inp_list]
    # if "ctx_ids_len" in inp_list[0]:
    #     out["ctx_ids_len"] = [x.pop("ctx_ids_len") for x in inp_list]
    return out


def generation_collator(inp_list, tokenizer):
    padding_kwargs = dict(padding=True, padding_side="left", return_tensors="pt")
    input_ids = [x.pop("input_ids") for x in inp_list]
    attn_mask = [x.pop("attention_mask") for x in inp_list]
    labels = [x.pop("labels") for x in inp_list]
    for i, label in enumerate(labels):
        # remove the response tokens
        idx = np.argmax(label != -100)
        idx = max(1, idx)
        input_ids[i] = input_ids[i][:idx]
        attn_mask[i] = attn_mask[i][:idx]
    out = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attn_mask}, **padding_kwargs
    )

    # we don't include the labels in the output
    # since we don't want to compute the loss on the labels
    # during generation
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
