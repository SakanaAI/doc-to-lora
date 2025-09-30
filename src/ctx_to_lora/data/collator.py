import numpy as np
import torch
from transformers.data import (
    DataCollatorWithFlattening,
    default_data_collator,
)

from ctx_to_lora.utils import check_is_iterable, concat_list

flattener = DataCollatorWithFlattening()


def flatten_if_not_packed(inp_list):
    # no padding
    sample = inp_list[0]
    n = len(inp_list)
    # training data is packed
    if "position_ids" in sample:
        if n == 1:
            n_queries = sample.pop("n_queries")
            n_ctx_chunks = sample.pop("n_ctx_chunks")
            batch = default_data_collator(inp_list, return_tensors="pt")
            batch["n_queries"] = torch.tensor(n_queries)
            batch["n_ctx_chunks"] = torch.tensor(n_ctx_chunks)
            return batch
        elif n > 1:
            raise NotImplementedError("Please use batch_size=1 when using packed data")
            # when batch_size > 1 (never used?)
            # return default_data_collator(concat_batch(inp_list), return_tensors="pt")

    # for eval data (not packed) during training
    need_flatten = check_is_iterable(sample["input_ids"][0])
    assert not need_flatten, f"Validation data should not be nested."

    n_queries = torch.ones(len(inp_list), dtype=torch.int32)
    n_ctx_chunks = torch.tensor(
        [len(example["ctx_ids"]) for example in inp_list], dtype=torch.int32
    )
    packed_inputs = flattener(inp_list, return_tensors="pt")

    packed_inputs["n_queries"] = n_queries
    packed_inputs["n_ctx_chunks"] = n_ctx_chunks

    if "ctx_ids" in sample:
        # HACK: assumes 1 ctx chunk here
        # assert all(len(ctx_ids) == 1 for ctx_ids in sample["ctx_ids"]), (
        #     "ctx_ids can only have one chunk for eval. "
        #     "Please implement chunked ctx forward pass to handle this."
        # )
        ctx_ids = concat_list([example.pop("ctx_ids") for example in inp_list])
        ctx_position_ids = torch.cat([torch.arange(len(ids)) for ids in ctx_ids])
        ctx_ids = torch.tensor(concat_list(ctx_ids))

        packed_inputs["ctx_ids"] = ctx_ids.unsqueeze(0)
        packed_inputs["ctx_position_ids"] = ctx_position_ids.unsqueeze(0)
        # for eval info
        if "ctx_ids_len" in sample:
            packed_inputs["ctx_ids_len"] = [
                example["ctx_ids_len"] for example in inp_list
            ]

    return packed_inputs


def eval_collator(inp_list, tokenizer):
    # only used for teacher-forcing eval
    # input is a list of tokenized sequences
    padding_kwargs = dict(padding=True, padding_side="right", return_tensors="pt")

    has_ctx_ids = "ctx_ids" in inp_list[0]
    if has_ctx_ids:
        # pad to the longest ctx_len in the batch
        # which can have a different length from the input_ids, attn_mask, labels
        ctx_ids = [example.pop("ctx_ids") for example in inp_list]
        ctx_attn_mask = [torch.ones_like(x) for x in ctx_ids]
        ctx_ids = torch.nn.utils.rnn.pad_sequence(
            ctx_ids,
            batch_first=True,
            padding_value=0,
        )
        ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
            ctx_attn_mask,
            batch_first=True,
            padding_value=0,
        )

    for inp in inp_list:
        inp["attention_mask"] = torch.ones_like(inp["input_ids"])

    labels = [x.pop("labels") for x in inp_list]
    # need to pass the whole inp bc we also track the lengths (with specal keys)
    padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

    # hacky explicit padding since the labels are not padded by default
    labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels}

    if has_ctx_ids:
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask

    return out


def generation_collator(inp_list, tokenizer):
    padding_kwargs = dict(padding=True, padding_side="left", return_tensors="pt")
    input_ids = [torch.tensor(x.pop("input_ids")) for x in inp_list]
    labels = [x.pop("labels") for x in inp_list]
    for i, label in enumerate(labels):
        # we don't include the labels in the output during generation
        # remove the response tokens
        idx = np.argmax(label != -100)
        idx = max(1, idx)
        input_ids[i] = input_ids[i][:idx]
    attn_mask = [torch.ones_like(x) for x in input_ids]

    out = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attn_mask}, **padding_kwargs
    )

    if "ctx_ids" in inp_list[0]:
        # pad to the longest ctx_len in the batch
        # which can have a different length from the input_ids, attn_mask, labels
        ctx_ids = [example.pop("ctx_ids") for example in inp_list]
        n_chunks = [len(x) for x in ctx_ids]
        ctx_ids = concat_list(ctx_ids)
        ctx_ids = [torch.tensor(x) for x in ctx_ids]
        ctx_attn_mask = [torch.ones_like(x) for x in ctx_ids]

        ctx_ids = torch.nn.utils.rnn.pad_sequence(
            ctx_ids,
            batch_first=True,
            padding_value=0,
        )
        ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
            ctx_attn_mask,
            batch_first=True,
            padding_value=0,
        )
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask

        out["n_ctx_chunks"] = torch.tensor(n_chunks, dtype=torch.int32)
    return out
