from huggingface_hub import snapshot_download

if __name__ == "__main__":
    fw_dir = "./data/raw_datasets/self_gen/"
    snapshot_download(
        "Rujikorn/self_gen_qa_logprobs",
        repo_type="dataset",
        local_dir=fw_dir,
    )
