import torch
from datasets import load_dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    HfArgumentParser,
    TrainingArguments,
    DataCollatorForSeq2Seq,
)

from modeling_utils import ModulatedPretrainedModel
from training_utils import train_model


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
    parser = HfArgumentParser((TrainingArguments,))
    training_args, *_ = parser.parse_args_into_dataclasses()

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

    model = AutoModelForCausalLM.from_pretrained(
        "meta-llama/Llama-3.1-8B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    if isinstance(model, ModulatedPretrainedModel):

        def tokenize(example):
            model_inputs = tokenizer(
                example["prompt"],
                truncation=True,
                padding=False,
            )
            model_inputs["ctx_ids"] = tokenizer(example["context"]).input_ids
            model_inputs["ctx_attention_mask"] = tokenizer(example["context"]).attention_mask
            model_inputs["labels"] = ...
            return model_inputs

    else:

        def tokenize(example):
            inp = [
                ctx + "\n" + prompt for ctx, prompt in zip(example["context"], example["prompt"])
            ]
            model_inputs = tokenizer(
                inp,
                example["answer"],
                # add_special_tokens=True, ???
                truncation=True,
                padding=False,
            )

            input_ids = model_inputs["input_ids"]
            labels = [None] * len(input_ids)

            for i in range(len(input_ids)):
                sequence_ids = model_inputs.sequence_ids(i)
                labels[i] = [
                    -100 if sequence_id == 0 else label
                    for sequence_id, label in zip(sequence_ids, input_ids[i])
                ]
            model_inputs["labels"] = labels
            return model_inputs

    print("Loading dataset...")
    train_file = "../data/raw_datasets/context_numbers/train.jsonl"
    eval_file = "../data/raw_datasets/context_numbers/val.jsonl"
    dataset = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    train_ds = dataset["train"].map(tokenize, batched=True)
    eval_ds = {
        "train": dataset["train"].select(range(100)).map(tokenize, batched=True),
        "val": dataset["eval"].map(tokenize, batched=True),
    }

    # DataCollatorForSeq2Seq also pads the `labels`
    # useful when we're computing the labels manually
    # or masking the loss only on completion
    data_collator = DataCollatorForSeq2Seq(tokenizer, model=model, pad_to_multiple_of=8)

    train_model(
        model,
        train_ds,
        eval_ds,
        training_args,
        data_collator,
        compute_metrics,
    )


if __name__ == "__main__":
    main()
