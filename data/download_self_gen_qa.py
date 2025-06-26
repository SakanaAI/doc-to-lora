from huggingface_hub import snapshot_download

if __name__ == "__main__":
    fw_dir = "./data/raw_datasets/self_gen/"
    snapshot_download(
        "SakanaAI/self_gen_qa",
        repo_type="dataset",
        local_dir=fw_dir,
    )
