import os

from datasets import load_dataset


def process_sample(sample):
    """Process a single sample and return the processed data or None if invalid."""
    ctx = " ".join(
        [
            token
            for is_html, token in zip(
                sample["document"]["tokens"]["is_html"],
                sample["document"]["tokens"]["token"],
            )
            if not is_html
        ]
    ).strip()

    if not ctx:
        return {"prompt": None, "context": None, "answer": None}

    answers = []
    for answer in sample["annotations"]["short_answers"]:
        answers += answer["text"]

    if not answers:
        return {"prompt": None, "context": None, "answer": None}

    answer = answers[0].strip()
    if not answer:
        return {"prompt": None, "context": None, "answer": None}

    prompt = sample["question"]["text"].capitalize().strip() + " ?"
    return {"prompt": prompt, "context": ctx, "answer": answer.capitalize()}


def shift_contexts(examples):
    """Shift contexts by one position to create negative examples."""
    contexts = examples["context"]
    shifted_contexts = contexts[1:] + contexts[:1]
    examples["context"] = shifted_contexts
    return examples


if __name__ == "__main__":
    ds = load_dataset("google-research-datasets/natural_questions")
    ds = ds.shuffle(seed=42)

    for split in ["validation", "train"]:
        # Process samples in parallel using map
        processed_ds = ds[split].map(
            process_sample, remove_columns=ds[split].column_names, num_proc=16
        )

        # Filter out None values
        processed_ds = processed_ds.filter(lambda x: x["context"] is not None)

        # Shift contexts to create negative examples
        processed_ds = processed_ds.map(
            shift_contexts, batched=True, batch_size=len(processed_ds)
        )

        print(processed_ds)
        save_dir = "data/raw_datasets/negative_natural_questions"
        os.makedirs(save_dir, exist_ok=True)
        processed_ds.to_json(f"{save_dir}/{split}.jsonl")
