from huggingface_hub import snapshot_download

if __name__ == "__main__":
    fw_dir = "./data/raw_datasets/fw_qa_v2/"
    snapshot_download(
        "SakanaAI/fineweb_qa",
        repo_type="dataset",
        local_dir=fw_dir,
    )
