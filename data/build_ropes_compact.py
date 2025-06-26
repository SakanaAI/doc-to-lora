import gc

from datasets import Dataset, load_dataset
from tqdm import tqdm

if __name__ == "__main__":
    ds_name = "allenai/ropes"

    for split in ["train", "validation"]:
        ctx_qa_dict = dict()
        ds = load_dataset(ds_name, split=split)
        print(f"Original size: {len(ds)}")
        for i, sample in tqdm(enumerate(ds)):
            ctx_template = "{background}\n{situation}"
            response = sample["answers"]["text"][0]
            bg_txt = sample["background"]
            situation_txt = sample["situation"]
            ctx = ctx_template.format(background=bg_txt, situation=situation_txt)
            q = sample["question"]
            if ctx not in ctx_qa_dict:
                ctx_qa_dict[ctx] = {"prompts": [], "responses": []}
            ctx_qa_dict[ctx]["prompts"].append(q)
            ctx_qa_dict[ctx]["responses"].append(response)

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

        save_path = f"./data/raw_datasets/ropes_compact/{split}/ds.parquet"
        print(f"Saving dataset to {save_path}")
        ds.to_parquet(save_path)
        print("=" * 80)
        del ds, samples, ctx_qa_dict
        gc.collect()
