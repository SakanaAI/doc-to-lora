import transformers
from peft import (
    LoraConfig,
)
from datasets import load_dataset
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
from training_utils import pretrain_tokenize_function, DataCollatorForDynamicPadding, train_model

def main():
    parser = transformers.HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)
    
    training_args.gradient_checkpointing_kwargs = {"use_reentrant": False}  # manually add this argument in the code

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM"
    )
    
    # check model_args.mem_size and min_tokens_for_lm
    assert (training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)) == 0, "training_args.fixed_mem_size must be a power of 2"    
    assert training_args.leave_tokens_for_lm <= training_args.min_tokens_for_lm, "leave_tokens_for_lm should be fewer than min_tokens_for_lm"

    
    memory_size = training_args.fixed_mem_size

    train_file = "/path/to/train/file"
    eval_file = "/path/to/dev/file"

    print("Loading dataset...")

    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file}, streaming=True) # streaming can be removed if the dataset is not very large.
    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    model = ICAE(model_args, training_args, lora_config)
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    train_dataset = train_dataset.map(pretrain_tokenize_function, batched=True, batch_size=64, fn_kwargs={"model": model, "mem": MEM_TOKENS, "lm_ratio": training_args.lm_ratio})
    eval_dataset = eval_dataset.map(pretrain_tokenize_function, batched=True, fn_kwargs={"model": model, "mem": MEM_TOKENS})   # don't add lm in the dev set.

    data_collator = DataCollatorForDynamicPadding(model.pad_token_id)
    train_model(model, train_dataset, eval_dataset, training_args, data_collator)

main()