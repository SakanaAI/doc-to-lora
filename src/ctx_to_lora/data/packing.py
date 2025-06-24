# based on
# https://github.com/MeetKai/functionary/blob/aa3dbdd65f7e388f2386622606bdfeec95c2b863/functionary/train/packing/packed_dataset.py
import logging

import numpy as np

from ctx_to_lora.utils import check_is_iterable, concat_list

logger = logging.getLogger()


def pack_data_points_by_length(
    lens: list[list[int]],
    ctx_lens: list[int],
    n_queries: list[int],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_size: int = -1,
) -> list[int]:
    if not lens:
        return []

    len_arr = np.array([sum(l) for l in lens], dtype=np.int32)
    ctx_len_arr = np.array(ctx_lens, dtype=np.int32)
    n = len(len_arr)
    assert len(ctx_len_arr) == n, "Length of ctx_lens must match length of lens"
    assert len(n_queries) == n, "Length of n_queries must match length of lens"

    if n == 1:
        return (
            [[0]]
            if len_arr[0] <= max_packed_inp_len or ctx_len_arr[0] <= max_packed_ctx_len
            else []
        )

    # Create cumulative sum arrays for efficient range queries
    cumsum_inp_len = np.cumsum(len_arr)
    cumsum_ctx_len = np.cumsum(ctx_len_arr)
    cumsum_n_queries = np.cumsum(n_queries)  # the last one is not used

    start_indices_inp = [0]
    start_indices_ctx = [0]
    i = 0

    while i < n:
        # Find the maximum j such that sum(lens[i:j+1]) <= max_packed_inp_len
        start_sum_inp = cumsum_inp_len[i - 1] if i > 0 else 0
        valid_ends_inp = cumsum_inp_len[i:] - start_sum_inp <= max_packed_inp_len

        start_sum_ctx = cumsum_ctx_len[i - 1] if i > 0 else 0
        valid_ends_ctx = cumsum_ctx_len[i:] - start_sum_ctx <= max_packed_ctx_len
        valid_ends = valid_ends_inp & valid_ends_ctx

        # this should never happen?
        if not np.any(valid_ends):
            # Single item exceeds max_packed_inp_len, skip it
            i += 1
            continue

        # Find the last valid index
        max_valid_idx = i + np.where(valid_ends)[0][-1]

        # Apply max_size constraint
        if max_size != -1:
            max_valid_idx = min(max_valid_idx, i + max_size - 1)

        start_indices_inp.append(cumsum_n_queries[max_valid_idx])
        start_indices_ctx.append(max_valid_idx + 1)
        i = max_valid_idx + 1

    return start_indices_inp, start_indices_ctx


def pack_data_points_FA(
    batch: dict[str, any],
) -> dict[str, np.ndarray]:
    if not batch:
        raise ValueError("Batch is empty")

    # Pre-allocate lists with known sizes
    total_ctx_len = sum(len(x) for x in batch["ctx_ids"])
    total_inp_len = sum(len(x) for x in batch["input_ids"])

    ctx_ids = np.empty(total_ctx_len, dtype=np.long)
    ctx_position_ids = np.empty(total_ctx_len, dtype=np.long)
    input_ids = np.empty(total_inp_len, dtype=np.long)
    position_ids = np.empty(total_inp_len, dtype=np.long)
    labels = np.empty(total_inp_len, dtype=np.long)

    offset = ctx_offset = 0
    for input_ids_b, labels_b in zip(batch["input_ids"], batch["labels"]):
        inp_len = len(input_ids_b)

        inp_start, inp_end = offset, offset + inp_len

        input_ids[inp_start:inp_end] = input_ids_b
        position_ids[inp_start:inp_end] = np.arange(inp_len, dtype=np.long)
        labels[inp_start:inp_end] = labels_b
        offset += inp_len

    for ctx_ids_b in batch["ctx_ids"]:
        ctx_len = len(ctx_ids_b)
        ctx_start, ctx_end = ctx_offset, ctx_offset + ctx_len
        ctx_ids[ctx_start:ctx_end] = ctx_ids_b
        ctx_position_ids[ctx_start:ctx_end] = np.arange(ctx_len, dtype=np.long)
        ctx_offset += ctx_len

    return {
        "ctx_ids": ctx_ids,
        "ctx_position_ids": ctx_position_ids,
        "input_ids": input_ids,
        "position_ids": position_ids,
        "labels": labels,
    }


def pack_batch(
    batch: dict[str, any],
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int = -1,
) -> dict[str, any]:
    # TODO: handle new data format
    # Extract lengths
    # we do not pad so we can just take the length of the tokens
    keys = list(batch.keys())
    need_flatten = check_is_iterable(batch["input_ids"][0][0])
    if need_flatten:
        n_queries = [len(x) for x in batch["input_ids"]]
        inp_lens = [[len(y) for y in x] for x in batch["input_ids"]]
        # batch = {k: concat_list([x for x in batch[k]]) for k in keys}
        for k in ["input_ids", "attention_mask", "labels"]:
            batch[k] = concat_list([x for x in batch[k]])
    else:
        n_queries = [1] * len(batch["input_ids"])
        inp_lens = [[len(x)] for x in batch["input_ids"]]
    inp_count = len(inp_lens)
    # total_original_tokens = sum(inp_lens)
    if "ctx_ids" not in batch:
        raise ValueError("Batch must contain 'ctx_ids' and 'labels' keys")
    # we do not pad so we can just take the length of the tokens
    ctx_lens = [len(x) for x in batch["ctx_ids"]]

    # Group indices
    start_indices_inp, start_indices_ctx = pack_data_points_by_length(
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

    packing_efficiency_ratios = []
    ctx_packing_efficiency_ratios = []

    for i in range(len(start_indices_inp) - 1):
        start_idx_inp = start_indices_inp[i]
        end_idx_inp = start_indices_inp[i + 1]
        start_idx_ctx = start_indices_ctx[i]
        end_idx_ctx = start_indices_ctx[i + 1]
        group_items = {
            "ctx_ids": batch["ctx_ids"][start_idx_ctx:end_idx_ctx],
            "input_ids": batch["input_ids"][start_idx_inp:end_idx_inp],
            "labels": batch["labels"][start_idx_inp:end_idx_inp],
        }
        packed_item = pack_data_points_FA(group_items)
        packed_batch["ctx_ids"].append(packed_item["ctx_ids"])
        packed_batch["ctx_position_ids"].append(packed_item["ctx_position_ids"])
        packed_batch["input_ids"].append(packed_item["input_ids"])
        packed_batch["position_ids"].append(packed_item["position_ids"])
        packed_batch["labels"].append(packed_item["labels"])
        packed_batch["n_queries"].append(n_queries[start_idx_ctx:end_idx_ctx])

    #     # Track packing efficiency
    #     group_inp_tokens = sum(inp_lens[start_idx:end_idx])
    #     group_ctx_tokens = sum(ctx_lens[start_idx:end_idx])

    #     if max_packed_inp_len > 0:
    #         inp_efficiency = group_inp_tokens / max_packed_inp_len
    #         packing_efficiency_ratios.append(inp_efficiency)

    #     if max_packed_ctx_len > 0:
    #         ctx_efficiency = group_ctx_tokens / max_packed_ctx_len
    #         ctx_packing_efficiency_ratios.append(ctx_efficiency)

    # # Calculate total original tokens
    # total_original_inp_tokens = sum(inp_lens)
    # total_original_ctx_tokens = sum(ctx_lens)
    # total_original_tokens = total_original_inp_tokens + total_original_ctx_tokens

    # # Calculate length statistics
    # packed_inp_lens_arr = np.array([len(x) for x in packed_batch["input_ids"]])
    # packed_ctx_lens_arr = np.array([len(x) for x in packed_batch["ctx_ids"]])

    # # Log performance statistics
    # compression_ratio = inp_count / n_samples if n_samples > 0 else 0
    # avg_inp_packing_efficiency = (
    #     np.mean(packing_efficiency_ratios) if packing_efficiency_ratios else 0
    # )
    # avg_ctx_packing_efficiency = (
    #     np.mean(ctx_packing_efficiency_ratios) if ctx_packing_efficiency_ratios else 0
    # )

    # print(
    #     f"Packing stats - Original samples: {inp_count}\n"
    #     f"Packed samples: {n_samples}\n"
    #     f"Compression ratio (sample-wise): {compression_ratio:.2f}x\n\n"
    #     f"Total inp tokens: {total_original_inp_tokens}\n"
    #     f"Total ctx tokens: {total_original_ctx_tokens}\n"
    #     f"Avg inp packing efficiency: {avg_inp_packing_efficiency:.3f}\n"
    #     f"Avg ctx packing efficiency: {avg_ctx_packing_efficiency:.3f}\n\n"
    #     f"Input IDs length stats:\n"
    #     f"  Avg: {np.mean(packed_inp_lens_arr):.1f}, Std: {np.std(packed_inp_lens_arr):.1f}, "
    #     f"Min: {np.min(packed_inp_lens_arr)}, Max: {np.max(packed_inp_lens_arr)}\n"
    #     f"Context IDs length stats:\n"
    #     f"  Avg: {np.mean(packed_ctx_lens_arr):.1f}, Std: {np.std(packed_ctx_lens_arr):.1f}, "
    #     f"Min: {np.min(packed_ctx_lens_arr)}, Max: {np.max(packed_ctx_lens_arr)}"
    # )

    return packed_batch


if __name__ == "__main__":
    from datasets import disable_caching

    from ctx_to_lora.data.processing import get_tokenized_dataset
    from ctx_to_lora.model_loading import get_model_and_tokenizer

    disable_caching()

    model, tokenizer = get_model_and_tokenizer(
        "google/gemma-2-2b-it",
        train=True,
        requires_grad=False,
    )
    base_model_max_len = model.base_model.config.max_position_embeddings

    tokenizer_kwargs = {"max_length": base_model_max_len}  # not used
    ctx_tokenizer_kwargs = {"max_length": base_model_max_len}  # not used for now
    ds = get_tokenized_dataset(
        ds_name="fw_qa_v2_2k_len_level_0_tiny",
        split="train",
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
    packed_ds = ds.map(
        pack_batch,
        fn_kwargs={
            "max_packed_inp_len": base_model_max_len * 2,
            "max_packed_ctx_len": base_model_max_len * 4,
        },
        batched=True,
        batch_size=1000,
        remove_columns=ds.column_names,
        # num_proc=0,
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
