from datasets import load_dataset
from modeling_icae_multi_span import (
    ICAE,
    DataArguments,
    ModelArguments,
    TrainingArguments,
)
from peft import LoraConfig
from training_utils import (
    DataCollatorForDynamicPadding,
    instruct_ft_tokenize_function,
    train_model,
)
from transformers import HfArgumentParser


def compute_metrics(eval_pred) -> dict:
    """
    Custom metrics function for the trainer
    Args:
        eval_pred: tuple of predictions and labels
    Returns:
        dictionary containing metric names (str) and values (Any)
    """

    # preds, labels = eval_preds

    # # predictions is generated tokens for Seq2SeqTrainer
    # # decode preds and labels
    # labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    # decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    # decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

    # compute per token accuracy
    predictions, labels = eval_pred.predictions, eval_pred.label_ids

    # predictions is logits for Trainer
    preds = predictions.argmax(-1)
    acc = (preds == labels).mean()
    return {"per_token_acc": acc}


def main():
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    print(model_args)
    print(data_args)

    training_args.eval_on_start = True
    training_args.eval_strategy = "steps"
    training_args.eval_steps = 500
    training_args.save_strategy = "no"
    # training_args.save_steps = 500
    training_args.logging_strategy = "steps"
    training_args.logging_steps = 100

    # seq2seq args for generation evaluation
    # training_args.predict_with_generate = True
    # training_args.generation_max_length = 100

    training_args.gradient_checkpointing_kwargs = {
        "use_reentrant": False
    }  # manually add this argument in the code

    lora_config = LoraConfig(
        r=model_args.lora_r,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )

    # check model_args.mem_size and min_tokens_for_lm
    assert (
        training_args.fixed_mem_size & (training_args.fixed_mem_size - 1)
    ) == 0, "training_args.fixed_mem_size must be a power of 2"

    memory_size = training_args.fixed_mem_size

    train_file = "../data/raw_datasets/context_numbers/train.jsonl"
    eval_file = "../data/raw_datasets/context_numbers/val.jsonl"

    print("Loading dataset...")

    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})

    train_dataset = dataset["train"]
    eval_dataset = dataset["eval"]

    model = ICAE(model_args, training_args, lora_config).to("cuda")
    MEM_TOKENS = list(range(model.vocab_size, model.vocab_size + memory_size))

    tokenized_train_ds = train_dataset.map(
        instruct_ft_tokenize_function,
        batched=True,
        fn_kwargs={"model": model, "mem": MEM_TOKENS},
    )

    tokenized_eval_ds = {"train": train_dataset.select(range(100)), "val": eval_dataset}

    for split in tokenized_eval_ds:
        tokenized_eval_ds[split] = tokenized_eval_ds[split].map(
            instruct_ft_tokenize_function,
            batched=True,
            fn_kwargs={"model": model, "mem": MEM_TOKENS},
        )

    train_model(
        model,
        tokenized_train_ds,
        tokenized_eval_ds,
        training_args,
        compute_metrics=compute_metrics,
    )


if __name__ == "__main__":
    main()
