import json
import random
from typing import List, Dict
import os


def save_jsonl(data: List[Dict], filepath: str) -> None:
    """Save data to a JSONL file."""
    parent_dir = os.path.dirname(filepath)
    if parent_dir:  # Only create directories if there's a parent path
        os.makedirs(parent_dir, exist_ok=True)
    with open(filepath, "w") as f:
        for entry in data:
            json.dump(entry, f)
            f.write("\n")


def generate_number_dataset(max_num: int = 1000):
    """
    Generate a dataset of numbers with corresponding query and answer,
    split into train/val/test sets.

    Args:
        max_num: Maximum number in the range (exclusive)
    """
    # Generate list of all numbers and shuffle them
    numbers = list(range(max_num))
    random.shuffle(numbers)

    # Create dataset entries
    dataset = []
    query = "What's your favourite number in [0-999]? Answer with only the number."

    for num in numbers:
        entry = {"context": str(num), "prompt": query, "response": str(num)}
        dataset.append(entry)

    # Calculate split sizes
    total_size = len(dataset)
    train_size = int(0.9 * total_size)
    val_size = int(0.05 * total_size)

    # Split dataset
    train_data = dataset[:train_size]
    val_data = dataset[train_size : train_size + val_size]
    test_data = dataset[train_size + val_size :]

    # Save splits to separate files
    save_jsonl(train_data, "train.jsonl")
    save_jsonl(val_data, "val.jsonl")
    save_jsonl(test_data, "test.jsonl")


if __name__ == "__main__":
    # Set random seed for reproducibility
    random.seed(42)

    # Generate dataset
    generate_number_dataset()

    print(f"Dataset splits generated and saved.")
