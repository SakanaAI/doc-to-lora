# based on
# https://github.com/MeetKai/functionary/blob/aa3dbdd65f7e388f2386622606bdfeec95c2b863/functionary/train/packing/packed_dataset.py
import json
import logging
import os
import pprint

import numpy as np

from ctx_to_lora.utils import check_is_iterable, concat_list, setup_logging

logger = logging.getLogger()


def pack_data_points_by_length(
    lens: list[list[int]],
    ctx_lens: list[int],
    n_queries: list[int],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_size: int = -1,
) -> tuple[list[int], list[int]]:
    if not lens:
        return []

    len_arr = np.array([sum(l) for l in lens], dtype=np.long)
    ctx_len_arr = np.array(ctx_lens, dtype=np.long)
    n = len(len_arr)
    assert len(ctx_len_arr) == n, "Length of ctx_len_arr must match length of lens"
    assert len(n_queries) == n, "Length of n_queries must match length of lens"

    if n == 1:
        return (
            [0]
            if len_arr[0] <= max_packed_inp_len and ctx_len_arr[0] <= max_packed_ctx_len
            else []
        )

    # Create cumulative sum arrays for efficient range queries
    cumsum_inp_len = np.cumsum(len_arr)
    cumsum_ctx_len = np.cumsum(ctx_len_arr)

    # indices of flattened input_ids
    cumsum_n_queries = np.cumsum(n_queries)

    # start_indices_inp = [0]
    # start_indices_ctx = [0]
    idx_pairs = []
    i = 0

    while i < n:
        # Find the maximum j such that sum(lens[i:j+1]) <= max_packed_inp_len
        start_sum_inp = cumsum_inp_len[i - 1] if i > 0 else 0
        valid_ends_inp = (cumsum_inp_len[i:] - start_sum_inp) <= max_packed_inp_len

        start_sum_ctx = cumsum_ctx_len[i - 1] if i > 0 else 0
        valid_ends_ctx = (cumsum_ctx_len[i:] - start_sum_ctx) <= max_packed_ctx_len
        valid_ends = valid_ends_inp & valid_ends_ctx

        if not np.any(valid_ends):
            # Single item exceeds max_packed_inp_len, skip it
            logging.debug(
                f"Skipping item {i} with input length {len_arr[i]} and context length {ctx_len_arr[i]}"
            )
            i += 1
            continue

        # Find the last valid index
        max_valid_idx = i + np.where(valid_ends)[0][-1]

        # Apply max_size constraint
        if max_size > 0:
            max_valid_idx = min(max_valid_idx, i + max_size - 1)

        idx_pairs.append((i, max_valid_idx + 1))
        i = max_valid_idx + 1

    return idx_pairs


def pack_data_points_FA(
    batch: dict[str, any],
) -> dict[str, np.ndarray]:
    if not batch:
        raise ValueError("Batch is empty")

    # Pre-allocate lists with known sizes
    total_ctx_len = sum(len(x) for x in batch["ctx_ids"])
    total_inp_len = sum(len(y) for x in batch["input_ids"] for y in x)

    ctx_ids = np.empty(total_ctx_len, dtype=np.long)
    ctx_position_ids = np.empty(total_ctx_len, dtype=np.long)
    input_ids = np.empty(total_inp_len, dtype=np.long)
    position_ids = np.empty(total_inp_len, dtype=np.long)
    labels = np.empty(total_inp_len, dtype=np.long)

    has_logprobs = "logprobs_vals" in batch

    if has_logprobs:
        sequences = zip(
            batch["input_ids"],
            batch["labels"],
            batch["logprobs_vals"],
            batch["logprobs_indices"],
        )
        n_labels = sum(len(y) for x in batch["logprobs_vals"] for y in x)
        k = len(batch["logprobs_vals"][0][0][0])  # assuming all have same k
        logprobs_vals = np.empty((n_labels, k), dtype=np.float32)
        logprobs_indices = np.empty((n_labels, k), dtype=np.int32)
        logits_offset = 0
    else:
        sequences = zip(batch["input_ids"], batch["labels"])

    offset = ctx_offset = 0
    for sample in sequences:
        input_ids_b, labels_b = sample[:2]
        inp_start = offset

        # compute position_ids for each sub-list in input_ids_b
        local_start = inp_start
        for ids_b in input_ids_b:
            position_ids[local_start : local_start + len(ids_b)] = np.arange(
                len(ids_b), dtype=np.long
            )
            local_start += len(ids_b)

        input_ids_b = concat_list(input_ids_b)
        labels_b = concat_list(labels_b)

        inp_len = len(input_ids_b)
        inp_end = offset + inp_len

        input_ids[inp_start:inp_end] = input_ids_b
        labels[inp_start:inp_end] = labels_b
        offset += inp_len

        if has_logprobs:
            logprobs_vals_b, logprobs_indices_b = sample[2:]
            logprobs_vals_b = concat_list(logprobs_vals_b)
            logprobs_indices_b = concat_list(logprobs_indices_b)
            logits_len = len(logprobs_vals_b)
            logprobs_vals[logits_offset : logits_offset + logits_len] = logprobs_vals_b
            logprobs_indices[logits_offset : logits_offset + logits_len] = (
                logprobs_indices_b
            )
            logits_offset += logits_len

    for ctx_ids_b in batch["ctx_ids"]:
        ctx_len = len(ctx_ids_b)
        ctx_start, ctx_end = ctx_offset, ctx_offset + ctx_len
        ctx_ids[ctx_start:ctx_end] = ctx_ids_b
        ctx_position_ids[ctx_start:ctx_end] = np.arange(ctx_len, dtype=np.long)
        ctx_offset += ctx_len

    out = {
        "ctx_ids": ctx_ids,
        "ctx_position_ids": ctx_position_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }
    if has_logprobs:
        out["logprobs_vals"] = logprobs_vals
        out["logprobs_indices"] = logprobs_indices
    return out


def pack_batch(
    batch: dict[str, any],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int = -1,
    metadata_path: str = "",
) -> dict[str, any]:
    need_flatten = check_is_iterable(batch["input_ids"][0][0])
    assert need_flatten, (
        f"Packing requires the input_ids to be nested "
        f"(allowing multiple QAs per sample), but got {batch['input_ids'][0]}"
    )

    n_queries = [len(x) for x in batch["input_ids"]]
    inp_lens = [[len(y) for y in x] for x in batch["input_ids"]]
    inp_count = len(inp_lens)
    if "ctx_ids" not in batch:
        raise ValueError("Batch must contain 'ctx_ids' and 'labels' keys")
    # we do not pad so we can just take the length of the tokens
    ctx_lens = [len(x) for x in batch["ctx_ids"]]

    # Group indices
    idx_pairs = pack_data_points_by_length(
        inp_lens,
        ctx_lens,
        n_queries,
        max_packed_inp_len,
        max_packed_ctx_len,
        max_packed_size,
    )

    # Pack groups
    # n_samples = len(boundaries) - 1
    packed_batch = {
        "ctx_ids": [],
        "ctx_position_ids": [],
        "input_ids": [],
        "position_ids": [],
        "labels": [],
        "n_queries": [],
    }
    has_logprobs = "logprobs_vals" in batch
    if has_logprobs:
        packed_batch["logprobs_vals"] = []
        packed_batch["logprobs_indices"] = []

    packing_efficiency_ratios = []
    ctx_packing_efficiency_ratios = []

    for idx_pair in idx_pairs:
        start_idx, end_idx = idx_pair[0], idx_pair[1]
        # start_idx = start_indices[i]
        # end_idx = start_indices[i + 1]
        group_items = {
            "ctx_ids": batch["ctx_ids"][start_idx:end_idx],
            "input_ids": batch["input_ids"][start_idx:end_idx],
            "labels": batch["labels"][start_idx:end_idx],
        }
        if has_logprobs:
            group_items["logprobs_vals"] = batch["logprobs_vals"][start_idx:end_idx]
            group_items["logprobs_indices"] = batch["logprobs_indices"][
                start_idx:end_idx
            ]
        packed_item = pack_data_points_FA(group_items)
        packed_batch["ctx_ids"].append(packed_item["ctx_ids"])
        packed_batch["ctx_position_ids"].append(packed_item["ctx_position_ids"])
        packed_batch["input_ids"].append(packed_item["input_ids"])
        packed_batch["position_ids"].append(packed_item["position_ids"])
        packed_batch["labels"].append(packed_item["labels"])
        packed_batch["n_queries"].append(n_queries[start_idx:end_idx])
        if has_logprobs:
            packed_batch["logprobs_vals"].append(packed_item["logprobs_vals"])
            packed_batch["logprobs_indices"].append(packed_item["logprobs_indices"])

        if max_packed_inp_len > 0:
            inp_efficiency = len(packed_item["input_ids"]) / max_packed_inp_len
            packing_efficiency_ratios.append(inp_efficiency)

        if max_packed_ctx_len > 0:
            ctx_efficiency = len(packed_item["ctx_ids"]) / max_packed_ctx_len
            ctx_packing_efficiency_ratios.append(ctx_efficiency)

    # # Calculate total original tokens
    # total_original_inp_tokens = sum(inp_lens)
    # total_original_ctx_tokens = sum(ctx_lens)
    # total_original_tokens = total_original_inp_tokens + total_original_ctx_tokens

    # # Calculate length statistics
    packed_inp_lens_arr = np.array([len(x) for x in packed_batch["input_ids"]])
    packed_ctx_lens_arr = np.array([len(x) for x in packed_batch["ctx_ids"]])

    # # Log performance statistics
    avg_inp_packing_efficiency = (
        np.mean(packing_efficiency_ratios) if packing_efficiency_ratios else 0
    )
    avg_ctx_packing_efficiency = (
        np.mean(ctx_packing_efficiency_ratios) if ctx_packing_efficiency_ratios else 0
    )

    # Create packing statistics dictionary
    packing_stats = {
        "original_samples": inp_count,
        "packed_samples": len(idx_pairs),
        "avg_inp_packing_efficiency": float(avg_inp_packing_efficiency),
        "avg_ctx_packing_efficiency": float(avg_ctx_packing_efficiency),
        "input_ids_length_stats": {
            "avg": float(np.mean(packed_inp_lens_arr)),
            "std": float(np.std(packed_inp_lens_arr)),
            "min": int(np.min(packed_inp_lens_arr)),
            "max": int(np.max(packed_inp_lens_arr)),
        },
        "context_ids_length_stats": {
            "avg": float(np.mean(packed_ctx_lens_arr)),
            "std": float(np.std(packed_ctx_lens_arr)),
            "min": int(np.min(packed_ctx_lens_arr)),
            "max": int(np.max(packed_ctx_lens_arr)),
        },
    }

    # Save to metadata_path if provided
    if metadata_path:
        os.makedirs(os.path.dirname(metadata_path), exist_ok=True)
        with open(metadata_path, "w") as f:
            json.dump(packing_stats, f, indent=4)

    logging.debug(f"Packing stats:\n{pprint.pformat(packing_stats, indent=2)}")

    return packed_batch


if __name__ == "__main__":
    from datasets import disable_caching

    from ctx_to_lora.data.processing import get_tokenized_dataset
    from ctx_to_lora.model_loading import get_model_and_tokenizer

    setup_logging("tmp/packing_debug.log", debug=True)
    logger.info("Starting packing script...")
    model, tokenizer = get_model_and_tokenizer(
        "google/gemma-2-2b-it",
        train=True,
        requires_grad=False,
    )
    base_model_max_len = model.base_model.config.max_position_embeddings

    tokenizer_kwargs = {"max_length": base_model_max_len}  # not used
    ctx_tokenizer_kwargs = {"max_length": base_model_max_len}  # not used for now
    ds = get_tokenized_dataset(
        ds_name="self_gen/google/gemma-2-2b-it_temp_0.0/fw_qa_v2_2k_len_level_2_tiny",
        split="train",
        max_qas_len=2048,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        tokenizer_kwargs={},
        ctx_model_max_len=model.base_model.config.max_position_embeddings,
        ctx_tokenizer=tokenizer,
        ctx_tokenizer_kwargs={},
        add_ctx_to_chat=False,
        add_repeat_prompt=False,
        repeat_prob=0.0,
        add_negative_prompt=False,
        use_kl_loss=False,
    )
    # ds.set_format("torch")
    print(ds)
    disable_caching()

    packed_ds = ds.map(
        pack_batch,
        fn_kwargs={
            "max_packed_inp_len": 2**13,  # 8k
            "max_packed_ctx_len": 2**14,  # 16k
        },
        batched=True,
        batch_size=100_000,
        remove_columns=ds.column_names,
        num_proc=4,
    )
    print(packed_ds)

    # packed_ds = PackedDataset(
    #     dataset=ds,
    #     tokenizer=tokenizer,
    #     max_input_length=base_model_max_len,
    #     max_packed_length=base_model_max_len * 2,
    #     # max_packed_size=3,  # maximum number of data points being packed
    # )
    # packed_ds.stat()
    # packed_ds = Dataset.from_list(packed_ds)
    # packed_ds.set_format("torch")
    # print(packed_ds)
    # print(ds[0])

    # print(packed_ds[0])
    orig_seq = ds[0]["ctx_ids"]
    packed_seq = packed_ds[0]["ctx_ids"][: len(orig_seq)]
    print(orig_seq, packed_seq)

    print(orig_seq == packed_seq)
