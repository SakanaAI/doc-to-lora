import argparse

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctx_to_lora.data.processing import load_and_process_dataset


def add_logits(row, model, tok, device, args):
    ctx = row["context"]
    qs = row["prompts"]
    ans = row["responses"]
    responses_masks = []
    all_logits_values = []
    all_logits_positions = []

    for question, answer in zip(qs, ans):
        messages = [
            {
                "role": "user",
                "content": f"Answer the question based on the following information:\n{ctx}.\n\n{question}",
            },
            {"role": "assistant", "content": answer},
        ]
        all_tokens = tok.apply_chat_template(
            messages,
            tokenize=True,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            add_generation_prompt=False,
            return_dict=True,
        )

        input = torch.tensor([all_tokens["input_ids"]], device=device)
        with torch.no_grad():
            logits = model(input).logits

        top_logits, top_logits_positions = torch.topk(
            logits, args.max_logits_to_store, dim=-1
        )

        # Convert logits to specified precision
        dtype_map = {
            "float32": torch.float32,
            "bfloat16": torch.bfloat16,
            "float16": torch.float16,
        }
        target_dtype = dtype_map[args.logit_precision]
        top_logits = top_logits.to(target_dtype)

        tokens_to_mask = tok.apply_chat_template(
            [messages[0]],  # Just the user message
            tokenize=True,
            add_special_tokens=False,
            truncation=False,
            padding=False,
            add_generation_prompt=True,
            return_dict=True,
        )
        number_of_tokens_to_mask = len(tokens_to_mask["input_ids"])

        # Create mask: False for prompt tokens, True for response tokens
        responses_mask = torch.zeros(len(all_tokens["input_ids"]), dtype=torch.bool)
        responses_mask[number_of_tokens_to_mask:] = True

        responses_masks.append(responses_mask)
        all_logits_values.append(top_logits.cpu())
        all_logits_positions.append(top_logits_positions.cpu())

    row["labels_logits_values"] = all_logits_values
    row["labels_logits_positions"] = all_logits_positions
    row["responses_mask"] = responses_masks
    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ds_name", required=True, help="name of the dataset")
    ap.add_argument("--split", required=True, help="split of the dataset")
    ap.add_argument(
        "--max_logits_to_store",
        type=int,
        default=100,
        help="max number of logits kept per completion",
    )
    ap.add_argument(
        "--limit_samples",
        type=int,
        default=None,
        help="process only the first n dataset rows",
    )
    ap.add_argument(
        "--logit_precision",
        default="bfloat16",
        choices=["float32", "bfloat16", "float16"],
        help="dtype for saved logits",
    )
    ap.add_argument(
        "--model_name", required=True, help="model name to use for generating logits"
    )
    ap.add_argument(
        "--output_path", required=True, help="path to save the processed dataset"
    )
    args = ap.parse_args()

    # Load model and tokenizer
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(args.model_name)
    model = model.to(device)
    model.eval()
    tok = AutoTokenizer.from_pretrained(args.model_name)

    # Load dataset
    ds = load_and_process_dataset(
        args.ds_name,
        args.split,
        add_negative_prompt=False,
        add_repeat_prompt=False,
        repeat_prob=0,
        is_pretrain=False,
        streaming=False,
        num_proc=8,
    )

    # Limit samples if specified
    if args.limit_samples:
        ds = ds.select(range(min(args.limit_samples, len(ds))))

    # Process dataset with logits
    ds = ds.map(
        lambda row: add_logits(row, model, tok, device, args),
        batched=False,
        desc="Adding logits to dataset",
    )

    # Save the processed dataset
    ds.save_to_disk(args.output_path)
    print(f"Dataset saved to {args.output_path}")


if __name__ == "__main__":
    main()
