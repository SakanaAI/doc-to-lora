from huggingface_hub import snapshot_download

if __name__ == "__main__":
    self_gen_data_dir = "./data/raw_datasets/self_gen/"
    snapshot_download(
        "SakanaAI/SakanaAI/self_gen_qa_d2l",
        repo_type="dataset",
        local_dir=self_gen_data_dir,
    )
