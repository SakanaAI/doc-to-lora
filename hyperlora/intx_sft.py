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
from data_utils import (
    convert_ctx_prompt_response_to_messages,
    get_preprocessing_fn,
    get_sft_prompt_formatting_fn,
    tokenize_chat_messages,
)
from training_utils import TRAINING_TASK, train_model


def compute_metrics(eval_pred) -> dict:
    """
    Custom metrics function for the trainer
    Args:
        eval_pred: tuple of predictions and labels
    Returns:
        dictionary containing metric names (str) and values (Any)
    """
    # compute per token accuracy
    preds, labels = eval_pred.predictions, eval_pred.label_ids

    valid_mask = labels != -100
    valid_preds = preds[valid_mask].argmax(-1)
    valid_labels = labels[valid_mask]
    acc = (valid_preds == valid_labels).mean()
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
        # "meta-llama/Llama-3.2-1B-Instruct",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
    tokenizer.pad_token_id = tokenizer.eos_token_id
    tokenizer.padding_side = "right"

    max_seq_len = 1024

    # if isinstance(model, ModulatedPretrainedModel):

    #     # TODO: how to tokenize properly?
    #     def tokenize(example):
    #         model_inputs = tokenizer(
    #             example["prompt"],
    #             truncation=True,
    #             padding=False,
    #         )
    #         model_inputs["ctx_ids"] = tokenizer(example["context"]).input_ids
    #         model_inputs["ctx_attention_mask"] = tokenizer(
    #             example["context"]
    #         ).attention_mask
    #         model_inputs["labels"] = ...
    #         return model_inputs

    # else:

    #     def tokenize(example):
    #         inp = [
    #             ctx + "\n" + prompt
    #             for ctx, prompt in zip(example["context"], example["prompt"])
    #         ]
    #         model_inputs = tokenizer(
    #             inp,
    #             example["answer"],
    #             add_special_tokens=False,
    #             truncation=True,
    #             padding=False,
    #             max_length=max_seq_len,
    #         )

    #         input_ids = model_inputs["input_ids"]
    #         labels = [None] * len(input_ids)

    #         for i in range(len(input_ids)):
    #             sequence_ids = model_inputs.sequence_ids(i)
    #             labels[i] = [
    #                 -100 if sequence_id == 0 else label
    #                 for sequence_id, label in zip(sequence_ids, input_ids[i])
    #             ]
    #         model_inputs["labels"] = labels
    #         return model_inputs

    print("Loading dataset...")
    train_file = "../data/raw_datasets/context_numbers/train.jsonl"
    eval_file = "../data/raw_datasets/context_numbers/val.jsonl"
    ds = load_dataset("json", data_files={"train": train_file, "eval": eval_file})
    # preprocessing
    ds = ds.map(get_preprocessing_fn("context_numbers"))
    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)
    # for sft + chat_model, we need to convert the dataset to chat format
    # add "messages" field
    ds = ds.map(
        convert_ctx_prompt_response_to_messages,
        fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
    )
    # add "chat" field
    ds = ds.map(get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer))
    # tokenize the chat + mask the assistant inputs
    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={"tokenizer": tokenizer, "mask_assistant_inputs": True},
        remove_columns=ds["train"].column_names,
    )
    train_ds = tokenized_ds["train"]
    eval_ds = {
        "train": tokenized_ds["train"].select(range(100)),
        "val": tokenized_ds["eval"],
    }

    # train_ds = dataset["train"].map(tokenize, batched=True)
    # eval_ds = {
    #     "train": dataset["train"].select(range(100)).map(tokenize, batched=True),
    #     "val": dataset["eval"].map(tokenize, batched=True),
    # }

    # DataCollatorForSeq2Seq also pads the `labels`
    # useful when we're computing the labels manually
    # or masking the loss only on completion
    data_collator = DataCollatorForSeq2Seq(tokenizer, model, pad_to_multiple_of=8)

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
