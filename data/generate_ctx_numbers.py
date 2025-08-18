import json
import os
import random


def save_jsonl(data: list[dict], filepath: str) -> None:
    """Save data to a JSONL file."""
    parent_dir = os.path.dirname(filepath)
    if parent_dir:  # Only create directories if there's a parent path
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def get_random_combinations(numbers: list[int], n: int, k: int) -> list[list[int]]:
    """Get n random combinations of k numbers from the list."""
    # using itertools.combinations hangs with large numbers
    # so we explicitly generate the n indices
    indices = [random.sample(range(len(numbers)), k=k) for _ in range(n)]
    return [[numbers[i] for i in ind] for ind in indices]


def generate_number_dataset(max_num: int = 1000, n: int = 12000, k: int = 3):
    """
    Generate a dataset of numbers with corresponding query and answer,
    split into train/val/test sets.

    Args:
        max_num: Maximum number in the range (exclusive)
        k: Number of elements in each combination
    """
    # Generate list of all numbers and shuffle them
    numbers = list(range(max_num))
    random.shuffle(numbers)

    # Create dataset entries
    dataset = []
    query = f"Repeat the information above exactly. Do not output anything else."

    # Generate all unique combinations of k numbers
    combinations = get_random_combinations(numbers, n, k)
    # random.shuffle(combinations)

    for combination in combinations:
        num_list = [f"{i + 1}. {num}" for i, num in enumerate(combination)]
        entry = {
            "context": "\n".join(num_list),
            "prompt": query,
            "response": "\n".join(num_list),
        }
        dataset.append(entry)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.98 * total_size)
    val_size = int(0.01 * total_size)

    # Split dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]
    return train_data, val_data, test_data


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)
    tok_bins = [(64, 128), (128, 256), (256, 512)] + [
        (512 + 256 * i, 512 + 256 * (i + 1)) for i in range(14)
    ]
    # roughly 9 tokens per number
    tok_per_num = [9 if bin[0] >= 1024 else 8 for bin in tok_bins]
    len_bins = [
        (bin[0] // tok, bin[1] // tok) for bin, tok in zip(tok_bins, tok_per_num)
    ]

    for len_bin, tok_bin in zip(len_bins, tok_bins):
        bin_size = len_bin[1] - len_bin[0]
        save_dir = f"data/raw_datasets/ctx_numbers_{tok_bin[0]}_{tok_bin[1]}"
        train_data, val_data, test_data = [], [], []
        for k in range(*len_bin):
            train, val, test = generate_number_dataset(n=128_000 // bin_size, k=k)
            train_data += train
            val_data += val
            test_data += test

        # Save splits to separate files
        os.makedirs(save_dir, exist_ok=True)
        save_jsonl(train_data, f"{save_dir}/train.jsonl")
        save_jsonl(val_data, f"{save_dir}/val.jsonl")
        save_jsonl(test_data, f"{save_dir}/test.jsonl")

        print(f"Dataset generated and saved at {save_dir}")
