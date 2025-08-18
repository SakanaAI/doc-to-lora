import argparse
import json
import math
import os
import random

# -----------------------------
# Config knobs (edit or use CLI)
# -----------------------------
TOKENS_PER_BLOCK = 40  # rough heuristic tokens per noise block
BASE_SAMPLES_PER_BIN = 12_800
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


def generate_magic_dataset(n: int, k: int) -> tuple[list[dict], list[dict], list[dict]]:
    """Generate n samples for a given block length k, evenly distributed across 10 depth bins."""
    # Evenly divide n across 10 depth bins
    base = n // 10
    rem = n % 10
    counts = [base + (1 if i < rem else 0) for i in range(10)]

    dataset: list[dict] = []
    for depth_bin, c in enumerate(counts):
        for _ in range(c):
            dataset.append(_build_example(total_blocks=k, depth_bin=depth_bin))

    # Randomize before splitting to ensure val/test are random samples
    random.shuffle(dataset)

    # 98/1/1 split
    total = len(dataset)
    train_sz = int(0.98 * total)
    val_sz = int(0.01 * total)
    train = dataset[:train_sz]
    val = dataset[train_sz : train_sz + val_sz]
    test = dataset[train_sz + val_sz :]
    return train, val, test


def main():
    parser = argparse.ArgumentParser(
        description="Generate noise-wrapped special magic number dataset (similar structure to generate_ctx_kv.py)",
    )
    parser.add_argument("--seed", type=int, default=RNG_SEED, help="Random seed")
    parser.add_argument(
        "--base-samples-per-bin",
        type=int,
        default=BASE_SAMPLES_PER_BIN,
        help="Baseline number of samples per token bin (scaled by bin width)",
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

    # Same token bins pattern as generate_ctx_kv.py (current version)
    tok_bins = [(64, 128), (128, 256), (256, 512)] + [
        (512 + 256 * i, 512 + 256 * (i + 1)) for i in range(14)
    ]

    if args.only_first_n_bins is not None:
        tok_bins = tok_bins[: args.only_first_n_bins]

    # Map token bins to block-length bins using heuristic
    len_bins = [
        (lo // args.tokens_per_block, hi // args.tokens_per_block)
        for (lo, hi) in tok_bins
    ]

    if args.dry_run:
        k = max(1, len_bins[0][0] or 1)
        train, _, _ = generate_magic_dataset(n=10, k=k)
        print("Sample entry:")
        print(json.dumps(train[0], indent=2))
        return

    for len_bin, tok_bin in zip(len_bins, tok_bins):
        bin_size = max(1, len_bin[1] - len_bin[0])
        save_dir = f"{args.out_prefix}_{tok_bin[0]}_{tok_bin[1]}"
        train_data: list[dict] = []
        val_data: list[dict] = []
        test_data: list[dict] = []

        for k in range(max(1, len_bin[0]), max(1, len_bin[1])):
            per_k = max(1, args.base_samples_per_bin // bin_size)
            train, val, test = generate_magic_dataset(n=per_k, k=k)
            train_data += train
            val_data += val
            test_data += test

        os.makedirs(save_dir, exist_ok=True)
        save_jsonl(train_data, f"{save_dir}/train.jsonl")
        save_jsonl(val_data, f"{save_dir}/val.jsonl")
        save_jsonl(test_data, f"{save_dir}/test.jsonl")

        print(f"Dataset generated and saved at {save_dir}")


if __name__ == "__main__":
    main()
