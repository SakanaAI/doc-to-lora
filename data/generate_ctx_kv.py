import argparse
import json
import os
import random
import uuid
from collections.abc import Iterable

import wonderwords

# -----------------------------
# Simple, editable config knobs
# -----------------------------
# You can change these defaults or pass CLI flags: --key-type / --value-type
DEFAULT_KEY_TYPE = "uuid"  # choices: "uuid" | "digits" | "words"
DEFAULT_VALUE_TYPE = "uuid"  # choices: "uuid" | "digits" | "words"
PAIR_SEP = "\n"  # between pairs
KV_SEP = " : "  # between key and value
TOKENS_PER_PAIR = 10  # rough heuristic used for bucketing by context length
BASE_SAMPLES_PER_BIN = 128_000
RNG_SEED = 42

nouns = wonderwords.random_word._get_words_from_text_file("nounlist.txt")
adjs = wonderwords.random_word._get_words_from_text_file("adjectivelist.txt")
# verbs = wonderwords.random_word._get_words_from_text_file("verblist.txt")
words = [f"{adj}-{noun}" for adj in adjs for noun in nouns]
words = sorted(list(set(words)))


def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save data to a JSONL file."""
    parent_dir = os.path.dirname(filepath)
    if parent_dir:
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def _gen_uuid8() -> str:
    return uuid.uuid4().hex[:8]


def _gen_digits6() -> str:
    return f"{random.randint(0, 999_999):06d}"


def _make_generator(kind: str):
    kind = kind.lower()
    if kind == "uuid":
        return _gen_uuid8
    if kind == "digits":
        return _gen_digits6
    if kind == "words":
        return _gen_word
    raise ValueError(f"Unknown kind '{kind}', expected 'uuid', 'digits', or 'words'")


def _gen_digits4() -> str:
    return f"{random.randint(0, 9_999):04d}"


def _gen_word() -> str:
    return random.choice(words)


def _make_value_generator(kind: str):
    """Like _make_generator, but when kind is 'digits' use 4 digits for values."""
    kind = kind.lower()
    if kind == "uuid":
        return _gen_uuid8
    if kind == "digits":
        return _gen_digits4
    if kind == "words":
        return _gen_word
    raise ValueError(f"Unknown kind '{kind}', expected 'uuid', 'digits', or 'words'")


def _unique(seq: Iterable[str]) -> list[str]:
    s = set()
    out = []
    for x in seq:
        if x not in s:
            s.add(x)
            out.append(x)
    return out


def render_context(pairs: list[tuple[str, str]]) -> str:
    """Render key-value pairs as a single context string.

    Example: "k1:v1 | k2:v2 | k3:v3"
    """
    return PAIR_SEP.join([f"{k}{KV_SEP}{v}" for k, v in pairs])


def generate_kv_dataset(n: int, k: int, key_type: str, value_type: str):
    """Generate n examples. Each example has k key-value pairs in the context.

    - context: "k1:v1 | k2:v2 | ..."
    - prompt:  "What is the value of `{k}`?"
    - response: "v" (exactly the value corresponding to the chosen k)
    """
    key_gen = _make_generator(key_type)
    val_gen = _make_value_generator(value_type)

    dataset = []

    for _ in range(n):
        # Generate unique keys to avoid ambiguity
        keys: list[str] = []
        while len(keys) < k:
            keys = _unique([key_gen() for _ in range(k * 2)])  # oversample, then dedupe
        keys = keys[:k]
        values = [val_gen() for _ in range(k)]

        pairs = list(zip(keys, values))

        # Choose a random key from the context to ask about
        q_key = random.choice(keys)
        q_val = values[keys.index(q_key)]

        entry = {
            "context": f"key{KV_SEP}value\n" + render_context(pairs),
            "prompt": f'What is the value of key "{q_key}"? Reply with only the value.',
            "response": q_val,
        }
        dataset.append(entry)

    # Split into train/val/test matching the same structure as generate_ctx_numbers.py
    total_size = len(dataset)
    train_size = int(0.98 * total_size)
    val_size = int(0.01 * total_size)

    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]
    return train_data, val_data, test_data


def main():
    parser = argparse.ArgumentParser(
        description="Generate key-value context QA data with the same structure as generate_ctx_numbers.py",
    )
    parser.add_argument(
        "--key-type",
        choices=["uuid", "digits", "words"],
        default=DEFAULT_KEY_TYPE,
        help="Type of keys to generate: 8-char uuid, 6-digit numbers, or words",
    )
    parser.add_argument(
        "--value-type",
        choices=["uuid", "digits", "words"],
        default=DEFAULT_VALUE_TYPE,
        help="Type of values to generate: 8-char uuid, 4-digit numbers, or words",
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
        default="data/raw_datasets/ctx_kv",
        help="Output directory prefix (bin range will be appended)",
    )
    parser.add_argument(
        "--tokens-per-pair",
        type=int,
        default=TOKENS_PER_PAIR,
        help="Heuristic tokens per key:value pair for bucketing",
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

    # Same token bins as generate_ctx_numbers.py
    tok_bins = [(64, 128), (128, 256), (256, 512)] + [
        (512 + 256 * i, 512 + 256 * (i + 1)) for i in range(14)
    ]

    if args.only_first_n_bins is not None:
        tok_bins = tok_bins[: args.only_first_n_bins]

    # Map token bins to pair-length bins using a simple heuristic
    len_bins = [
        (lo // args.tokens_per_pair, hi // args.tokens_per_pair)
        for (lo, hi) in tok_bins
    ]

    # Optional dry-run: build a tiny sample to preview format and exit
    if args.dry_run:
        train, val, test = generate_kv_dataset(
            n=3,
            k=max(1, len_bins[0][0] or 1),
            key_type=args.key_type,
            value_type=args.value_type,
        )
        print("Sample entry:")
        print(json.dumps(train[0], indent=2))
        return

    for len_bin, tok_bin in zip(len_bins, tok_bins):
        bin_size = max(1, len_bin[1] - len_bin[0])  # avoid div-by-zero for tiny bins
        save_dir = f"{args.out_prefix}_{tok_bin[0]}_{tok_bin[1]}"
        train_data, val_data, test_data = [], [], []

        # Iterate over different context lengths (number of pairs)
        for k in range(max(1, len_bin[0]), max(1, len_bin[1])):
            # scale like the numbers script
            per_k = max(1, args.base_samples_per_bin // bin_size)
            train, val, test = generate_kv_dataset(
                n=per_k, k=k, key_type=args.key_type, value_type=args.value_type
            )
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
