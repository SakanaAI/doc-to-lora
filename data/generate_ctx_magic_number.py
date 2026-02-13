import argparse
import json
import math
import os
import random

# -----------------------------
# Config knobs (edit or use CLI)
# -----------------------------
TOKENS_PER_BLOCK = 40  # rough heuristic tokens per noise block
BASE_SAMPLES_PER_BIN = (
    320_000  # training samples budget scaler only (val/test fixed at 1000 each)
)
RNG_SEED = 42
NOISE_BLOCK = "The grass is green. The sky is blue. The sun is yellow. Here we go. There and back again."
SPECIAL_TPL = "The special magic number is {magic_number}."
SEP = "\n"  # between blocks


def save_jsonl(data: list[dict], filepath: str) -> None:
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


essential_digits4 = lambda: f"{random.randint(0, 9_999):04d}"


def _choose_position(total_blocks: int, depth_bin: int) -> int:
    """Choose an insertion index for the special sentence within [0, total_blocks-1]
    such that its relative depth falls within the depth bin [i/10, (i+1)/10).
    """
    if total_blocks <= 0:
        return 0
    # Use floor for start and ceil for end to cover boundaries evenly
    start = math.floor(total_blocks * (depth_bin / 10))
    end = math.ceil(total_blocks * ((depth_bin + 1) / 10)) - 1
    # clamp
    start = max(0, min(start, total_blocks - 1))
    end = max(start, min(end, total_blocks - 1))
    return random.randint(start, end)


def _build_example(total_blocks: int, depth_bin: int) -> dict:
    """Build one example with a special line inserted among noise blocks.

    total_blocks: total number of blocks in the final context (including the special one)
    depth_bin: integer in [0, 9]
    """
    total_blocks = max(1, total_blocks)

    # Prepare blocks
    magic = essential_digits4()
    special_line = SPECIAL_TPL.format(magic_number=magic)

    # We'll have (total_blocks - 1) noise blocks and 1 special line
    noise_count = max(0, total_blocks - 1)
    blocks: list[str] = [NOISE_BLOCK for _ in range(noise_count)]

    insert_at = _choose_position(total_blocks, depth_bin)
    # Insert special line at the desired position within the final sequence
    # If noise_count == 0, we just return special
    if noise_count == 0:
        final_blocks = [special_line]
    else:
        # Compose by interleaving noise and inserting special at index
        # Build a list of length `total_blocks` and fill
        final_blocks = []
        noise_idx = 0
        for idx in range(total_blocks):
            if idx == insert_at:
                final_blocks.append(special_line)
            else:
                final_blocks.append(blocks[noise_idx])
                noise_idx += 1

    context = SEP.join(final_blocks)
    prompt = "What is the special magic number? Reply with only the number."
    response = magic
    return {"context": context, "prompt": prompt, "response": response}


def generate_examples(n: int, k: int) -> list[dict]:
    """Generate n examples (all for block length k) evenly across 10 depth bins."""
    if n <= 0:
        return []
    base = n // 10
    rem = n % 10
    counts = [base + (1 if i < rem else 0) for i in range(10)]
    out: list[dict] = []
    for depth_bin, c in enumerate(counts):
        for _ in range(c):
            out.append(_build_example(total_blocks=k, depth_bin=depth_bin))
    random.shuffle(out)
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise-wrapped special magic number dataset (similar structure to generate_ctx_kv.py)",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")
    parser.add_argument(
        "--tokenizer-name",
        type=str,
        default="google/gemma-2-2b-it",
        help=("Tokenizer name"),
    )
    parser.add_argument(
        "--base-samples-per-bin",
        type=int,
        default=BASE_SAMPLES_PER_BIN,
        help="Baseline number of TRAINING samples per token bin (scaled by bin width). Validation & test are always 1000 each.",
    )
    parser.add_argument(
        "--out-prefix",
        type=str,
        default="data/raw_datasets/ctx_magic_number",
        help="Output directory prefix (bin range will be appended)",
    )
    parser.add_argument(
        "--tokens-per-block",
        "--tokens-per-pair",
        dest="tokens_per_block",
        type=int,
        default=TOKENS_PER_BLOCK,
        help="Heuristic tokens per noise block for bucketing",
    )
    parser.add_argument(
        "--only-first-n-bins",
        type=int,
        default=None,
        help="For quick tests: only generate the first N token bins",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print a small sample and exit without writing files",
    )

    args = parser.parse_args()

    random.seed(args.seed)

    # ----------------------------------------------------
    # Optional: report tokenizer-based token length stats
    # ----------------------------------------------------
    if args.tokenizer_name:
        try:
            from transformers import AutoTokenizer  # type: ignore
        except Exception as e:  # pragma: no cover
            raise RuntimeError(
                "Failed to import transformers. Install it or omit --tokenizer-name."
            ) from e

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name, use_fast=True)
        noise_token_count = len(tokenizer(NOISE_BLOCK).input_ids)
        special_example = SPECIAL_TPL.format(magic_number="0000")
        special_token_count = len(tokenizer(special_example).input_ids)
        print(
            f"[Tokenizer: {args.tokenizer_name}] Noise block tokens: {noise_token_count} | Special line tokens: {special_token_count}"
        )

    tok_bins = [(32, 128), (128, 256), (256, 512), (512, 1024), (32, 1024)] + [
        (1024 * i, 1024 * (i + 1)) for i in range(1, 16)
    ]
    tok_bins += [(2**14 + 2**12 * (i), 2**14 + 2**12 * (i + 1)) for i in range(4)]
    tok_bins += [(2**15 + 2**13 * (i), 2**15 + 2**13 * (i + 1)) for i in range(12)]
    if args.only_first_n_bins is not None:
        tok_bins = tok_bins[: args.only_first_n_bins]

    if args.tokenizer_name:
        max_hi = max(hi for _, hi in tok_bins)

        def measure_len(k: int) -> int:
            if k == 1:
                ctx = SPECIAL_TPL.format(magic_number="0000")
            else:
                blocks = [NOISE_BLOCK] * (k - 1) + [
                    SPECIAL_TPL.format(magic_number="0000")
                ]
                ctx = SEP.join(blocks)
            return len(tokenizer(ctx).input_ids)

        lengths: list[int] = [0]
        k = 1
        while True:
            L = measure_len(k)
            lengths.append(L)
            if L >= max_hi:
                break
            k += 1

        len_bins = []
        for lo, hi in tok_bins:
            k_lo = None
            for kk in range(1, len(lengths)):
                if lengths[kk] >= lo:
                    k_lo = kk
                    break
            if k_lo is None or lengths[k_lo] >= hi:
                len_bins.append((0, 0))
                continue
            k_hi = len(lengths)
            for kk in range(k_lo, len(lengths)):
                if lengths[kk] >= hi:
                    k_hi = kk
                    break
            len_bins.append((k_lo, k_hi))

        base_tokens = lengths[1]
        delta = (lengths[2] - lengths[1]) if len(lengths) > 2 else 0
        print(
            f"Using tokenizer-measured block ranges. base_tokens={base_tokens} approx_delta={delta}"
        )
    else:
        len_bins = [
            (lo // args.tokens_per_block, hi // args.tokens_per_block)
            for (lo, hi) in tok_bins
        ]

    if args.dry_run:
        for lb in len_bins:
            if lb[1] > lb[0]:
                k = max(1, lb[0])
                sample = generate_examples(10, k)
                print("Sample entry:")
                print(json.dumps(sample[0], indent=2))
                break
        return
    # -----------------------------------------------
    # Main generation per token bin
    # -----------------------------------------------
    TARGET_VAL = 1000
    TARGET_TEST = 1000
    for len_bin, tok_bin in zip(len_bins, tok_bins):
        if len_bin[1] <= len_bin[0]:
            print(f"Skipping token bin {tok_bin} (no valid block counts)")
            continue
        k_start = max(1, len_bin[0])
        k_end = max(1, len_bin[1])
        k_values = list(range(k_start, k_end))
        bin_size = len(k_values)
        save_dir = f"{args.out_prefix}_{tok_bin[0]}_{tok_bin[1]}"
        training_enabled = tok_bin[1] <= 1024  # unchanged policy
        if training_enabled:
            train_data: list[dict] = []
            # Distribute training budget across k values.
            # Scale: per_k = base_samples_per_bin / bin_size
            per_k_train = max(1, args.base_samples_per_bin // max(1, bin_size))
            for k in k_values:
                train_data += generate_examples(per_k_train, k)
        val_data: list[dict] = []
        test_data: list[dict] = []
        base_val = TARGET_VAL // bin_size
        rem_val = TARGET_VAL % bin_size
        base_test = TARGET_TEST // bin_size
        rem_test = TARGET_TEST % bin_size
        for idx, k in enumerate(k_values):
            n_val_k = base_val + (1 if idx < rem_val else 0)
            n_test_k = base_test + (1 if idx < rem_test else 0)
            if n_val_k:
                val_data += generate_examples(n_val_k, k)
            if n_test_k:
                test_data += generate_examples(n_test_k, k)
        random.shuffle(val_data)
        random.shuffle(test_data)
        os.makedirs(save_dir, exist_ok=True)
        if training_enabled:
            save_jsonl(train_data, f"{save_dir}/train.jsonl")
        save_jsonl(val_data, f"{save_dir}/val.jsonl")
        save_jsonl(test_data, f"{save_dir}/test.jsonl")
        if training_enabled:
            print(
                f"Dataset generated at {save_dir} (train={len(train_data)} val={len(val_data)} test={len(test_data)})"
            )
        else:
            print(
                f"Dataset (val/test only) generated at {save_dir} (val={len(val_data)} test={len(test_data)})"
            )


if __name__ == "__main__":
    main()
