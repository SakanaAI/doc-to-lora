import os

from datasets import Dataset, load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    ds = load_dataset("google-research-datasets/natural_questions")
    ds = ds.shuffle(seed=42)
    for split in ["train", "validation"]:
        out = []
        for i, sample in enumerate(tqdm(ds[split])):
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
                continue
            answers = []
            for answer in sample["annotations"]["short_answers"]:
                answers += answer["text"]
            if not answers:
                continue
            answer = answers[0].strip()
            if not answer:
                continue
            prompt = sample["question"]["text"].capitalize().strip() + " ?"
            out.append(dict(prompt=prompt, context=ctx, answer=answer.capitalize()))
            # if i >= 10:
            #     break
        shifted_ctxs = [d["context"] for d in out]
        shifted_ctxs = shifted_ctxs[1:] + shifted_ctxs[:1]
        for d, shifted_ctx in zip(out, shifted_ctxs):
            d["context"] = shifted_ctx
        new_ds = Dataset.from_list(out)
        print(new_ds)
        save_dir = "data/raw_datasets/negative_natural_questions"
        os.makedirs(save_dir, exist_ok=True)
        new_ds.to_json(f"{save_dir}/{split}.jsonl")
