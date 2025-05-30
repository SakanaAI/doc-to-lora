import gc
from collections import defaultdict
from glob import glob

from datasets import Dataset, load_dataset
from tqdm import tqdm

QA_TEMPLATE = "\n\nQuestion: {question}\nAnswer: {answer}"

if __name__ == "__main__":
    root_data_dir = "./data/raw_datasets/fw_qa_3"
    files = glob(f"{root_data_dir}/*.parquet")
    for file in files:
        ctx_qa_dict = defaultdict(str)
        ds = load_dataset("parquet", data_files=file, split="train")
        print(f"Loading dataset from {file}")
        print(f"Original size: {len(ds)}")
        for i, sample in tqdm(enumerate(ds)):
            ctx = sample["context"]
            question = sample["prompt"]
            answer = sample["response"]
            ctx_qa_dict[ctx] += QA_TEMPLATE.format(question=question, answer=answer)
        print(f"Unique contexts: {len(ctx_qa_dict)}")
        sampled_data = ctx_qa_dict[ctx]
        print(f"Sampled context-qa pairs: {ctx}{''.join(sampled_data)}")
        # convert ctx_qa_dict to a list of dictionaries
        samples = [
            {"context": ctx, "qas": qa_pairs} for ctx, qa_pairs in ctx_qa_dict.items()
        ]
        # save to a new dataset
        ds = Dataset.from_list(samples)
        save_path = f"./data/raw_datasets/fw_qa_intx_pretrain/{file.split('/')[-1]}"
        print(f"Saving dataset to {save_path}")
        ds.to_parquet(save_path)
        print("=" * 80)
        del ds
        del samples
        gc.collect()
