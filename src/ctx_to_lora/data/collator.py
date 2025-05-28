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
