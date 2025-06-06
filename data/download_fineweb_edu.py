from huggingface_hub import snapshot_download

if __name__ == "__main__":
    fw_dir = "./data/raw_datasets/fineweb_edu/"
    snapshot_download(
        "HuggingFaceFW/fineweb-edu",
        repo_type="dataset",
        local_dir=fw_dir,
        allow_patterns="sample/100BT/*",
    )
