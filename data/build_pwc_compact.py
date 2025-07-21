import gc

from datasets import Dataset, load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    ds_name = "sggetao/PwC"

    for split in ["train", "test"]:
        ctx_qa_dict = dict()
        ds = load_dataset(ds_name, split=split)
        print(f"Original size: {len(ds)}")
        for i, sample in tqdm(enumerate(ds)):
            ctx = sample["input"]
            if ctx not in ctx_qa_dict:
                ctx_qa_dict[ctx] = {"prompts": [], "responses": []}
            # question = closed_qa_prompting(sample["prompt"])
            question = sample["prompt"]
            answer = sample["answer"]
            ctx_qa_dict[ctx]["prompts"].append(question)
            ctx_qa_dict[ctx]["responses"].append(answer)

        print(f"Unique contexts: {len(ctx_qa_dict)}")
        # convert ctx_qa_dict to a list of dictionaries
        samples = [
            {
                "context": ctx,
                "prompts": ctx_qa_dict[ctx]["prompts"],
                "responses": ctx_qa_dict[ctx]["responses"],
            }
            for ctx in ctx_qa_dict
        ]
        print(f"Sampled data: {samples[0]}")
        # breakpoint()
        # save to a new dataset
        ds = Dataset.from_list(samples)

        save_path = f"./data/raw_datasets/pwc_compact/{split}/ds.parquet"
        print(f"Saving dataset to {save_path}")
        ds.to_parquet(save_path)
        print("=" * 80)
        del ds, samples, ctx_qa_dict
        gc.collect()
