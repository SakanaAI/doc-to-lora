import random
from glob import glob

from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from transformers import set_seed


def remove_too_long(samples):
    return [len(text) < 10_000 for text in samples["text"]]


if __name__ == "__main__":
    set_seed(42)
    fw_dir = "./data/raw_datasets/fineweb/"
    snapshot_download(
        "HuggingFaceFW/fineweb",
        repo_type="dataset",
        local_dir=fw_dir,
        allow_patterns="sample/10BT/*",
    )
    # https://github.com/huggingface/datasets/issues/7047#issuecomment-2233163406
    num_shards_per_file = 16
    sharded_fw_dir = "./data/raw_datasets/fineweb_sharded/"
    output_path_template = f"{sharded_fw_dir}" + "/{i:02d}_{idx:05d}.parquet"

    for i, f in enumerate(sorted(glob(f"{fw_dir}/sample/10BT/*.parquet"))):
        # ~1M rows ~= 2GB mem required per file
        ds = Dataset.from_parquet(f)
        ds = ds.filter(remove_too_long, batched=True)
        print(f"Filtered ds size: {len(ds)}")
        ds = ds.shuffle(seed=42 + i)
        # take one shard (from 16 shards) per file
        # idx = random.sample(range(num_shards_per_file), 1)[0]
        # shard = ds.shard(index=idx, num_shards=num_shards_per_file, contiguous=False)
        # shard.to_parquet(output_path_template.format(index=i))
        for idx in range(num_shards_per_file):
            shard = ds.shard(index=idx, num_shards=num_shards_per_file, contiguous=True)
            shard.to_parquet(output_path_template.format(i=i, idx=idx))
