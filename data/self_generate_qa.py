import argparse
import os
import random
from glob import glob

import pandas as pd
import yaml
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from ctx_to_lora.data.definitions import RAW_DATA_DIR, SELF_GEN_DATA_DIR
from ctx_to_lora.data.processing import (
    filter_none,
    get_preprocessing_fn,
    load_and_process_dataset,
)
from ctx_to_lora.utils import clear_gpu

STOP_STRINGS = {
    "google/gemma-2-2b-it": ["<eos>", "<end_of_turn>"],
}

SYSTEM_TEMPLATE = (
    "### SYSTEM INSTRUCTIONS ###\n"
    "You are a creative and helpful assistant.\n"
    "**DO NOT** hallucinate or make up information.\n"
    "### END OF SYSTEM INSTRUCTIONS ###"
)

PROMPT_TEMPLATE = "### Context ###\n{context}\n\n\n### Question ###\n{question}"

MODEL_CTX_LEN = {
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
}


def get_prompt(context: str, q: str) -> str:
    return PROMPT_TEMPLATE.format(context=context, question=q)


def load_config(config_path: str) -> dict:
    """Load dataset names from YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


def check_should_skip(txt: str, vllm_model: str) -> bool:
    """Check if the response should be skipped based on stop strings."""
    for stop in STOP_STRINGS[vllm_model]:
        if stop in txt[-len(stop) :]:
            return (txt.split(stop)[0], False)  # Found a valid stop string
    return (txt, True)  # No valid stop string found, skip this response


def get_dataset_configs(
    ds_names: list[str] | None,
    config: dict | None,
    split: str | None,
) -> list[tuple[str, str]]:
    assert not (ds_names and config), "Cannot provide both ds_names and config"
    if ds_names:
        assert split, "When using ds_names, --split must be provided"
        # Validate ds_names format
        for ds_name in ds_names:
            if not isinstance(ds_name, str):
                raise ValueError(f"Invalid dataset name: {ds_name}")
        return [(ds_name, split) for ds_name in ds_names]

    if config:
        dataset_configs = []

        # Process train datasets
        train_ds_names = config.get("train_ds_names", [])
        self_gen_train_ds_names = [
            (ds_name.split("/")[-1], "train")
            for ds_name in train_ds_names
            if ds_name.startswith("self_gen/")
        ]
        if not self_gen_train_ds_names:
            print("No self_gen datasets found in train_ds_names")
        dataset_configs.extend(self_gen_train_ds_names)

        # Process validation datasets
        val_ds_names = config.get("val_ds_names", [])
        self_gen_val_ds_names = [
            (ds_name, "validation")
            for ds_name in val_ds_names
            if ds_name.startswith("self_gen/")
        ]
        if not self_gen_val_ds_names:
            print("No self_gen datasets found in val_ds_names")
        dataset_configs.extend(self_gen_val_ds_names)

        return dataset_configs


def create_messages(
    ctxs: list[str], questions: list[list[str]], vllm_model: str, system_template: str
) -> list[list[dict]]:
    """Create chat messages for the model."""
    if "gemma" in vllm_model:
        # gemma models do not support system messages
        return [
            [
                {
                    "role": "user",
                    "content": system_template + "\n\n\n" + get_prompt(ctx, q),
                }
            ]
            for ctx, q_list in zip(ctxs, questions)
            for q in q_list
        ]
    else:
        return [
            [
                {"role": "system", "content": system_template},
                {"role": "user", "content": get_prompt(ctx, q)},
            ]
            for ctx, q_list in zip(ctxs, questions)
            for q in q_list
        ]


def self_generate(
    ds_name: str,
    split: str,
    args: argparse.Namespace,
    llm: LLM,
    system_template: str,
    parquet_file: str | None = None,
) -> None:
    """Process a single dataset and generate QA pairs."""

    shard_name = ""
    kwargs = dict(
        ds_name=ds_name,
        split=split,
        add_negative_prompt=False,
        add_repeat_prompt=False,
        repeat_prob=0,
        is_pretrain=False,
    )
    if parquet_file:
        print(f"Loading dataset from parquet file: {parquet_file}")

        split = "train"
        ds_name = "/".join(parquet_file.split(RAW_DATA_DIR)[-1].split("/")[:-1])
        print(ds_name)
        shard_name = "_" + os.path.basename(parquet_file).replace(".parquet", "")
        ds = load_dataset(path="parquet", data_files=[parquet_file], split="train")
        processing_fn = get_preprocessing_fn(ds_name, is_eval=False, is_pretrain=False)
        ds = ds.map(processing_fn, num_proc=8)

    else:
        ds_name = ds_name.split("/")[-1]  # Extract just the dataset name

        print(f"Loading dataset: {ds_name} with split: {split}")

        ds = load_and_process_dataset(**kwargs, streaming=False, num_proc=8)

    temp = 0.0
    if "_temp_" in ds_name:
        temp = float(ds_name.split("_temp_")[-1].split("/")[0])
    if args.debug:
        ds = ds.take(10)

    ds = ds.filter(filter_none, batched=False, num_proc=8)

    ctxs = [sample["context"] for sample in ds]
    questions = [sample["prompts"] for sample in ds]
    print(f"Loaded {len(ctxs)} contexts and {len(questions)} questions")

    messages = create_messages(ctxs, questions, args.vllm_model, SYSTEM_TEMPLATE)

    print(f"Generating from {len(messages)} contexts")
    # TODO (distillation): make vllm outputs logits here too
    completions = llm.chat(
        messages,
        sampling_params=SamplingParams(
            max_tokens=1024,
            temperature=temp,
            # needed for checking if stop tokens are present
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        ),
    )

    self_gen_data = {ctx: {"prompts": [], "responses": []} for ctx in ctxs}
    c = 0
    n_skips = 0
    for ctx, q_list in zip(ctxs, questions):
        for i, q in enumerate(q_list):
            response = completions[c + i].outputs[0].text
            response, skip = check_should_skip(response, args.vllm_model)
            # skip = True
            # for stop in STOP_STRINGS[args.vllm_model]:
            #     if stop == response[-len(stop) :]:
            #         # Check if response ends with stop string
            #         response = response.split(stop)[0]
            #         skip = False
            #         break
            if skip:
                print(f"Skipping due to missing stop string")
                n_skips += 1
                continue

            self_gen_data[ctx]["prompts"].append(q)
            self_gen_data[ctx]["responses"].append(response)
        c += i + 1

    print(f"Skipped {n_skips} responses due to missing stop strings")
    samples = [
        {
            "context": ctx,
            "prompts": q_list,
            "responses": self_gen_data[ctx]["responses"],
        }
        for ctx, q_list in zip(ctxs, questions)
    ]

    if args.debug:
        for sample in samples:
            print(f"context={sample['context']}")
            print(f"prompt={sample['prompts']}")
            print(f"response={sample['responses']}")
            print("=" * 80)

    print(f"Generated {len(samples)} samples")
    random.shuffle(samples)

    # Save results
    if not args.debug:
        df = pd.DataFrame(samples)
        ds_out = Dataset.from_pandas(df)
        fpath = f"{SELF_GEN_DATA_DIR}/{args.vllm_model}_temp_{temp}/{ds_name}/{split}/ds{shard_name}"
        os.makedirs(os.path.dirname(fpath), exist_ok=True)
        fpath = f"{fpath}.parquet"
        ds_out.to_parquet(fpath)
        print(f"Saved to {fpath}")

        # Cleanup
        del samples, df, ds_out, completions, messages, ctxs, questions
        clear_gpu()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate QA pairs using VLLM")
    parser.add_argument(
        "--vllm_model",
        type=str,
        required=True,
        help="VLLM model name (e.g., google/gemma-2-2b-it)",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (process only 10 samples)",
    )

    # Either config file OR ds_names + split
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--config",
        type=str,
        help="Path to YAML config file with train_ds_names/val_ds_names",
    )
    group.add_argument(
        "--ds_names",
        type=str,
        nargs="+",
        help="List of dataset names/shard patterns",
    )
    group.add_argument(
        "--glob_pattern",
        type=str,
        help="Glob pattern to match dataset names (e.g., 'data/raw_datasets/fw_qa_3/*')",
    )

    parser.add_argument(
        "--split",
        type=str,
        help="Dataset split to use when using --ds_names (required with --ds_names)",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()

    # Validate arguments
    if args.ds_names and not args.split:
        raise ValueError("--split is required when using --ds_names")

    vllm_model = args.vllm_model
    print(f"Using model: {vllm_model}")

    # Setup model-specific configurations
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=MODEL_CTX_LEN.get(vllm_model),
    )

    print(f"{llm_kwargs=}")
    llm = LLM(**llm_kwargs)

    # Get dataset configs from config or CLI args
    config = load_config(args.config) if args.config else None
    if args.ds_names or args.config:
        dataset_configs = get_dataset_configs(
            ds_names=args.ds_names,
            config=config,
            split=args.split,
        )

        # Process each dataset
        for ds_name, split in dataset_configs:
            print(f"Processing dataset: {ds_name}, split: {split}")
            self_generate(ds_name, split, args, llm, SYSTEM_TEMPLATE)
    else:
        assert args.glob_pattern, (
            "glob_pattern must be provided if no ds_names or config"
        )
        files = glob(args.glob_pattern)
        for file in files:
            print(f"Processing file: {file}")
            self_generate(
                ds_name=None,
                parquet_file=file,
                split=args.split,
                args=args,
                llm=llm,
                system_template=SYSTEM_TEMPLATE,
            )
