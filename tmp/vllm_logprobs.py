from datasets import load_dataset

from ctx_to_lora.data.processing import get_tokenized_dataset
from ctx_to_lora.model_loading import get_tokenizer

if __name__ == "__main__":
    ds = load_dataset(
        "parquet",
        data_files="./data/raw_datasets/self_gen/google/gemma-2-2b-it_temp_0.0_closed_qa_prob_0.0/pwc_compact/train/ds.parquet",
        split="train",
    )

    # tokenizer = ctx_tokenizer = AutoTokenizer.from_pretrained(
    #     "google/gemma-2-2b-it",
    # )
    tokenizer = ctx_tokenizer = get_tokenizer("google/gemma-2-2b-it", train=True)

    tokenized_ds = get_tokenized_dataset(
        "self_gen/google/gemma-2-2b-it_temp_0.0_closed_qa_prob_0.0/pwc_compact",
        split="train",
        base_model_max_len=2**13,
        tokenizer=tokenizer,
        tokenizer_kwargs={},
        ctx_model_max_len=2**13,
        ctx_tokenizer=ctx_tokenizer,
        ctx_tokenizer_kwargs={},
        max_qas_len=2048,
        max_qas_per_sample=1,
        add_ctx_to_chat=False,
        add_repeat_prompt=False,
        add_negative_prompt=False,
        use_kl_loss=True,
    )
    print(ds[0])
    print(tokenized_ds[0])
    breakpoint()
