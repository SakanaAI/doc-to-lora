import argparse
import os
import random
import re
from glob import glob

import numpy as np
import yaml
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from ctx_to_lora.data.definitions import (
    CLOSED_QA_INTX_TEMPLATES,
    RAW_DATA_DIR,
    SELF_GEN_DATA_DIR,
)
from ctx_to_lora.data.processing import (
    filter_none,
    get_preprocessing_fn,
    load_and_process_dataset,
    tokenize_ctx_text,
)
from ctx_to_lora.data.self_gen_template import (
    PRE_CTX,
    PROMPT_TEMPLATE,
    QA_PROMPT_TEMPLATE,
    SELF_GEN_SYSTEM_MSG,
    SELF_QA_INTX,
)
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.utils import clear_gpu

STOP_STRINGS = {
    "google/gemma-2-2b-it": ["<eos>", "<end_of_turn>"],
}

MODEL_CTX_LEN = {
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
    # qwen 4b has 256k ctx length but using lower max lengths is faster
    "Qwen/Qwen3-4B-Instruct-2507": 2**13 + 2**12,
}


def truncate_middle_if_too_long(
    input_ids: list[int],
    max_length: int,
    max_new_tokens: int = 256,
) -> list[int]:
    """
    Truncate the middle of a list of tokens to fit within a maximum length.

    Args:
        tokens: List of token IDs
        max_length: Maximum length for the truncated tokens

    Returns:
        List of truncated token IDs
    """
    max_new_tokens_half = max_new_tokens // 2
    # leave max_new_tokens for generation
    half = max_length // 2 - max_new_tokens_half
    if len(input_ids) > max_length:
        return input_ids[:half] + input_ids[-half:]
    return input_ids


def get_prompt(context: str, q: str, remove_qa_template: bool) -> str:
    prompt = QA_PROMPT_TEMPLATE if not remove_qa_template else PROMPT_TEMPLATE
    return prompt.format(context=context, question=q)


def add_closed_qa_prompt(q: str, closed_qa_prob: float = 0.1) -> str:
    if random.random() <= closed_qa_prob:
        q = random.choice(CLOSED_QA_INTX_TEMPLATES).format(input=q)
    return q


def load_config(config_path: str) -> dict:
    """Load dataset names from YAML config file."""
    with open(config_path) as f:
        config = yaml.safe_load(f)
    return config


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
        # self_gen_train_ds_names = [
        #     (ds_name.split("/")[-1], "train")
        #     for ds_name in train_ds_names
        #     if ds_name.startswith("self_gen/")
        # ]
        self_gen_train_ds_names = [
            (ds_name, "train")
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
    ctxs: list[str],
    questions: list[list[str]],
    vllm_model: str,
    system_template: str,
    remove_qa_template: bool,
) -> list[list[dict]]:
    """Create chat messages for the model."""
    # if "gemma" in vllm_model:
    # gemma models do not support system messages
    return [
        [
            {
                "role": "user",
                "content": (
                    system_template + "\n\n\n" + get_prompt(ctx, q, remove_qa_template)
                ).strip(),
            }
        ]
        for ctx, q_list in zip(ctxs, questions)
        for q in q_list
    ]
    # else:
    #     return [
    #         [
    #             {"role": "system", "content": system_template},
    #             {"role": "user", "content": get_prompt(ctx, q)},
    #         ]
    #         for ctx, q_list in zip(ctxs, questions)
    #         for q in q_list
    #     ]


def self_generate(
    ds_name: str,
    split: str,
    args: argparse.Namespace,
    llm: LLM,
    system_template: str,
    parquet_file: str | None = None,
    do_truncate: bool = False,
) -> None:
    """Process a single dataset and generate QA pairs."""

    shard_name = ""

    # Conflict checks for ds_name-derived overrides
    if ds_name is not None:
        # temperature & closed_qa already handled later; add new ones
        if "_temp_" in ds_name and args.temp != 0.0:
            raise ValueError(
                f"Multiple sources of truth for temperature: CLI arg --temp={args.temp} and dataset name contains temp specification."
            )
        if "_closed_qa_prob_" in ds_name and args.closed_qa_prob != 0.0:
            raise ValueError(
                f"Multiple sources of truth for closed_qa_prob: CLI arg --closed_qa_prob={args.closed_qa_prob} and dataset name contains closed_qa_prob specification."
            )

    # Base values from args
    temp = args.temp
    closed_qa_prob = args.closed_qa_prob

    # Overrides from ds_name pattern if present
    if ds_name is not None:
        if "_temp_" in ds_name:
            m = re.search(r"_temp_([\d.]+)", ds_name)
            if m:
                temp = float(m.group(1))
        if "_closed_qa_prob_" in ds_name:
            m = re.search(r"_closed_qa_prob_([\d.]+)", ds_name)
            if m:
                closed_qa_prob = float(m.group(1))

    print(f"Processing dataset: {ds_name}, split: {split}")
    print(f"Using temperature: {temp}")
    print(f"Using closed QA prompt probability: {closed_qa_prob}")

    if parquet_file:
        print(f"Loading dataset from parquet file: {parquet_file}")

        split = "train"
        ds_name = "/".join(parquet_file.split(RAW_DATA_DIR)[-1].split("/")[:-1])

        shard_name = "_" + os.path.basename(parquet_file).replace(".parquet", "")
        ds = load_dataset(path="parquet", data_files=[parquet_file], split="train")
        processing_fn = get_preprocessing_fn(ds_name, is_eval=False)
        ds = ds.map(processing_fn, num_proc=8)

    else:
        ds_name = ds_name.split("/")[-1]  # Extract just the dataset name

        print(f"Loading dataset: {ds_name} with split: {split}")
        kwargs = dict(ds_name=ds_name, split=split)

        ds = load_and_process_dataset(**kwargs, num_proc=8, remove_cols=False)
    print(f"Loaded dataset: {ds_name} with split: {split}")

    if args.debug:
        ds = ds.take(10)

    ds = ds.filter(filter_none, batched=False, num_proc=8)

    tk = get_tokenizer(args.vllm_model, train=True)

    self_qa_intx_tokens = tk(SELF_QA_INTX, add_special_tokens=False)["input_ids"][1:]
    if args.remove_qa_template:
        self_qa_intx_tokens = tk("\n\n", add_special_tokens=False)["input_ids"]
    n_self_qa_intx_tokens = len(self_qa_intx_tokens)
    pre_ctx_tokens = tk(PRE_CTX, add_special_tokens=False)["input_ids"]
    n_pre_ctx_tokens = len(pre_ctx_tokens)
    sys_tokens = tk(system_template.split("\n")[0], add_special_tokens=False)[
        "input_ids"
    ][:-1]
    n_sys_tokens = len(sys_tokens)
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    ds = ds.map(
        tokenize_ctx_text,
        fn_kwargs={"tokenizer": tk},
        batched=True,
        batch_size=50_000,
        keep_in_memory=True,
    )

    ctxs = [sample["context"] for sample in ds]
    questions = [
        [add_closed_qa_prompt(q, closed_qa_prob) for q in sample["prompts"] if q]
        for sample in ds
    ]

    questions = [q_list for q_list in ds["prompts"] if len(q_list) > 0]

    print(f"Loaded {len(ctxs)} contexts and {len(questions)} questions")

    k = 16
    fpath = f"{SELF_GEN_DATA_DIR}/{args.vllm_model}_temp_{temp}_closed_qa_prob_{closed_qa_prob}/{ds_name}/{split}/ds{shard_name}"

    chunk_size = 1_000
    for chunk_idx, start in enumerate(range(0, len(ctxs), chunk_size)):
        print(f"Processing chunk {chunk_idx}")

        chunk_ctxs = ctxs[start : start + chunk_size]
        chunk_questions = questions[start : start + chunk_size]
        chunk_messages = create_messages(
            chunk_ctxs,
            chunk_questions,
            args.vllm_model,
            SELF_GEN_SYSTEM_MSG,
            args.remove_qa_template,
        )

        if do_truncate:
            # we should only do this for evaluation data
            tokenized_contents = tk(
                [m[0]["content"] for m in chunk_messages],
                add_special_tokens=False,
                return_attention_mask=False,
            )
            tokenized_contents["input_ids"] = [
                truncate_middle_if_too_long(
                    ids,
                    max_length=MODEL_CTX_LEN[args.vllm_model],
                    max_new_tokens=args.max_new_tokens,
                )
                for ids in tokenized_contents["input_ids"]
            ]
            contents = tk.batch_decode(
                tokenized_contents["input_ids"], skip_special_tokens=True
            )
            for c, m in zip(contents, chunk_messages):
                m[0]["content"] = c

        print(f"Generating from {len(chunk_messages)} contexts")

        # Clear GPU memory before processing the next chunk
        clear_gpu()
        execute_qa_generation(
            fpath + f"_{chunk_idx:04d}",
            args,
            llm,
            temp,
            tk,
            self_qa_intx_tokens,
            n_self_qa_intx_tokens,
            sys_tokens,
            n_sys_tokens,
            chunk_ctxs,
            ds[start : start + chunk_size]["ctx_ids"],
            chunk_questions,
            chunk_messages,
            k,
        )


def execute_qa_generation(
    fpath,
    args,
    llm,
    temp,
    tk,
    self_qa_intx_tokens,
    n_self_qa_intx_tokens,
    sys_tokens,
    n_sys_tokens,
    ctxs,
    ctx_ids,
    questions,
    messages,
    k,
):
    completions = llm.chat(
        messages,
        sampling_params=SamplingParams(
            max_tokens=args.max_new_tokens,
            logprobs=k,
            temperature=temp,
            seed=42,
            spaces_between_special_tokens=False,
            skip_special_tokens=False,
            include_stop_str_in_output=True,
        ),
    )

    self_gen_data = {
        ctx: {
            "ctx_ids": ctx_ids,
            "input_ids": [],
            "response_start_end": [],
            "logprobs_vals": [],
            "logprobs_indices": [],
        }
        for ctx, ctx_ids in zip(ctxs, ctx_ids)
    }
    c = 0
    n_skips = 0
    sys_start = None
    for ctx, q_list in zip(ctxs, questions):
        # self_gen_data[ctx]["ctx_ids"] = ctx_ids
        for i, _ in enumerate(q_list):
            # response = completions[c + i].outputs[0].text
            reason = completions[c + i].outputs[0].finish_reason
            if reason != "stop":
                # print(f"idx: {c + i}")
                print(f"finish_reason: {completions[c + i].outputs[0].finish_reason}")
                print(f"Skipping due to finish_reason={reason} != 'stop'")
                n_skips += 1
                continue

            # includes the logprob before the first response token
            # but excludes the logprob from eos token
            logp = completions[c + i].outputs[0].logprobs

            # len = num response tokens
            n_response_tokens = len(completions[c + i].outputs[0].token_ids)

            logp_indices = np.empty((n_response_tokens, k), dtype=np.int32)
            # float-16 is better for this range
            logp_vals = np.empty((n_response_tokens, k), dtype=np.float16)
            assert len(logp) == n_response_tokens, (
                f"Expected {n_response_tokens} logp entries, got {len(logp)}"
            )

            for li, info_d in enumerate(logp):
                for j, (idx, tok_info) in enumerate(info_d.items()):
                    logp_indices[li, j] = idx
                    logp_vals[li, j] = tok_info.logprob

            prompt_ids = completions[c + i].prompt_token_ids  # 1d list
            # token_ids only includes generated tokens, not the prompt
            response_token_ids = completions[c + i].outputs[0].token_ids  # 1d list
            all_ids = prompt_ids + response_token_ids
            res_start = len(prompt_ids)
            res_end = res_start + n_response_tokens

            if sys_start is None:
                for ii in range(len(prompt_ids) - n_sys_tokens):
                    if prompt_ids[ii : ii + n_sys_tokens] == sys_tokens:
                        # found the start of the system message
                        sys_start = ii
                        break

            q_start = None
            for ii in range(
                len(prompt_ids) - n_self_qa_intx_tokens,
                -1,
                -1,
            ):
                if prompt_ids[ii : ii + n_self_qa_intx_tokens] == self_qa_intx_tokens:
                    # found the start of the user input
                    q_start = ii + n_self_qa_intx_tokens
                    break

            # bos + question + eos + start model turn + response + eos
            input_ids = all_ids[:sys_start] + all_ids[q_start:res_end]

            # relative to the input_ids
            res_start = res_start - q_start + sys_start
            res_end = res_start + n_response_tokens

            # arrays will be saved as nested lists of numbers

            self_gen_data[ctx]["input_ids"].append(input_ids)
            # assume single-turn chat
            self_gen_data[ctx]["response_start_end"].append((res_start, res_end))
            self_gen_data[ctx]["logprobs_vals"].append(logp_vals)
            self_gen_data[ctx]["logprobs_indices"].append(logp_indices)

        c += i + 1

    print(f"Skipped {n_skips} responses due to missing stop strings")
    samples = [
        {
            # "context": ctx,
            # "prompts": q_list,
            # "responses": self_gen_data[ctx]["responses"],
            "ctx_ids": self_gen_data[ctx]["ctx_ids"],
            "input_ids": self_gen_data[ctx]["input_ids"],
            "response_start_end": self_gen_data[ctx]["response_start_end"],
            # "prompt_start_end": self_gen_data[ctx]["prompt_start_end"],
            "logprobs_vals": self_gen_data[ctx]["logprobs_vals"],
            "logprobs_indices": self_gen_data[ctx]["logprobs_indices"],
        }
        for ctx, q_list in zip(ctxs, questions)
    ]

    if args.debug:
        for sample in samples:
            # print(f"context={tk.decode(sample['ctx_ids'])}")
            print(f"QA={[tk.decode(ids) for ids in sample['input_ids']]}")

            for input_ids, (start, end) in zip(
                sample["input_ids"], sample["response_start_end"]
            ):
                print(f"start={start}, end={end}")
                print(f"response={tk.decode(input_ids[start:end])}")

            print(f"logprobs_vals={[x.shape for x in sample['logprobs_vals']]}")
            print(f"logprobs_indices={[x.shape for x in sample['logprobs_indices']]}")
            for indices in sample["logprobs_indices"]:
                print(f"logprobs_indices={indices[-1]}")
            print("=" * 80)

    print(f"Generated {len(samples)} samples")
    # random.shuffle(samples)

    # Save results
    # df = pd.DataFrame(samples)
    # ds_out = Dataset.from_pandas(df)
    ds_out = Dataset.from_list(samples)
    # fpath = f"{SELF_GEN_DATA_DIR}/{args.vllm_model}_temp_{temp}_closed_qa_prob_{closed_qa_prob}/{ds_name}/{split}/ds{shard_name}"

    if args.debug:
        fpath += "_debug"
    os.makedirs(os.path.dirname(fpath), exist_ok=True)

    fpath = f"{fpath}.parquet"
    ds_out.to_parquet(fpath)
    print(f"Saved to {fpath}")

    # Cleanup
    del samples, ds_out, completions, messages, ctxs, questions
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
    parser.add_argument(
        "--temp",
        type=float,
        default=0.0,
        help="Temperature for sampling (default: 0.0)",
    )
    parser.add_argument(
        "--closed_qa_prob",
        type=float,
        default=0.0,
        help="Probability of using closed QA prompt template (default: 0.0)",
    )
    parser.add_argument(
        "--do_truncate",
        action="store_true",
        help="Truncate contexts to fit model context length",
    )
    parser.add_argument(
        "--remove_qa_template",
        action="store_true",
        help="Remove QA template formatting from prompts",
    )
    parser.add_argument(
        "--max_new_tokens",
        type=int,
        default=256,
        help="Maximum number of new tokens to generate (default: 256)",
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
        max_num_batched_tokens=16384,
        max_num_seqs=32,  # avoid oom when getting logprobs
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
            self_generate(
                ds_name, split, args, llm, SELF_GEN_SYSTEM_MSG, None, args.do_truncate
            )
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
                system_template=SELF_GEN_SYSTEM_MSG,
                do_truncate=args.do_truncate,
            )
