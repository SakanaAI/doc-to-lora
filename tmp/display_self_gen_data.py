import sys

from datasets import load_dataset
from transformers import AutoTokenizer


def display_sample(ds, tk, i):
    sample = ds[i]
    print(f"ctx: {tk.decode(sample['ctx_ids'])}")
    for ids, (start, end) in zip(sample["input_ids"], sample["response_start_end"]):
        print(f"ids: {ids}")
        print(f"input: {tk.decode(ids)}")
        print(f"response: {tk.decode(ids[start:])}")
        len_input_ids = len(ids)
        pad_len_right = len_input_ids - end
        print(f"{pad_len_right=}")


data_files = sys.argv[1]

ds = load_dataset("parquet", data_files=data_files, split="train")
tk = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")

for i in range(5):
    display_sample(ds, tk, i)

breakpoint()
