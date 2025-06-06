import numpy as np
import torch
from transformers.data import DataCollatorWithFlattening

flattener = DataCollatorWithFlattening()


def train_packed_collator(inp_list):
    # no padding
    packed_inputs = flattener(inp_list, return_tensors="pt")
    if "ctx_ids" in inp_list[0]:
        ctx_ids = [{"input_ids": example["ctx_ids"]} for example in inp_list]
        packed_ctx = flattener(ctx_ids, return_tensors="pt")
        packed_inputs["ctx_ids"] = packed_ctx["input_ids"]
        packed_inputs["ctx_position_ids"] = packed_ctx["position_ids"]
        # for eval info
        if "ctx_ids_len" in inp_list[0]:
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
