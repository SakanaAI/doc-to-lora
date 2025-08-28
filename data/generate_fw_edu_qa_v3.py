import argparse
import os
import random
import re
from glob import glob

import pandas as pd
from datasets import Dataset, load_dataset
from q_generation_prompts import (  # get_structuring_seed_prompt,; get_summarization_seed_prompt,
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    get_creative_seed_prompt,
    get_generic_seed_prompt,
    get_question_seed_prompt,
    get_use_case_seed_prompt,
)
from vllm import LLM, SamplingParams

STOP_STRINGS = {
    "google/gemma-3-12b-it": ["<eos>", "<end_of_turn>"],
}


def build_messages(
    context: str,
    question_weight: float,
    use_case_weight: float,
    creative_weight: float,
    generic_weight: float,
) -> list[dict]:
    # Create list of instruction functions with their weights
    instruction_options = [
        ("question", get_question_seed_prompt, question_weight),
        ("use_case", get_use_case_seed_prompt, use_case_weight),
        ("creative", get_creative_seed_prompt, creative_weight),
        ("generic", get_generic_seed_prompt, generic_weight),
    ]

    q_types, functions, weights = zip(*instruction_options)

    chosen_idx = random.choices(range(len(instruction_options)), weights=weights)[0]

    q_type = q_types[chosen_idx]
    function = functions[chosen_idx]

    custom_instruction = function()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                context=context, instruction=custom_instruction
            ),
        },
    ]
    return messages, q_type


def check_should_skip(txt: str, vllm_model: str) -> bool:
    """Check if the response should be skipped based on stop strings."""
    for stop in STOP_STRINGS[vllm_model]:
        if stop in txt[-len(stop) :]:
            return (txt.split(stop)[0], False)  # Found a valid stop string
    return (txt, True)  # No valid stop string found, skip this response


def postprocess_questions(res_txt: str):
    """
    Postprocesses the questions from the response text.

    Args:
        res_txt: The response text.

    Returns:
        A list containing the questions.
    """
    # capture everything after each "Question {number}:" until the next question or end
    res_txt = res_txt.split("</think>")[-1]

    q_pattern = r"Message:(.*?)(?=Message \d+|$)"  # thanks chatgpt
    questions = re.findall(q_pattern, res_txt, flags=re.S)

    out_q = []
    n_skips = 0
    if len(questions) > 0:
        for i, question in enumerate(questions):
            question = question.strip()
            if not question:
                print(f"Skipping empty question at index {i}")
                continue
            # Check if this is the last question and handle stop strings
            if i == len(questions) - 1:
                question, skip = check_should_skip(question, vllm_model)
                if skip:
                    print(f"Skipping due to missing stop string")
                    n_skips += 1
                    continue
            out_q.append(question.strip())
    print(f"Skipped {n_skips} responses due to missing stop strings")

    return out_q


def length_filter(sample, min_len, max_len):
    return min_len <= len(sample["text"]) <= max_len


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from FineWeb Edu dataset"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=os.environ.get("vllm_model", "google/gemma-3-12b-it"),
        help="VLLM model to use for generation",
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        required=True,
        help="Pattern to match shard files (e.g., '000_0000*')",
    )
    # parser.add_argument(
    #     "--n_qa_pairs",
    #     type=int,
    #     required=True,
    #     help="Number of question-answer pairs to generate per context",
    # )
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum length of the context to consider for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=20_000,
        help="Maximum length of the context to consider for generation",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=2**13,
        help="Maximum length of the model input (context + prompt + response) in tokens",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode - process only first 100 samples",
    )
    parser.add_argument(
        "--question_weight",
        type=float,
        default=0,
        help="Weight for question seed prompts",
    )
    parser.add_argument(
        "--use_case_weight",
        type=float,
        default=0,
        help="Weight for use case seed prompts",
    )
    parser.add_argument(
        "--creative_weight",
        type=float,
        default=0,
        help="Weight for creative seed prompts",
    )
    parser.add_argument(
        "--generic_weight",
        type=float,
        default=0,
        help="Weight for generic seed prompts",
    )

    args = parser.parse_args()

    weights = [
        args.question_weight,
        args.use_case_weight,
        args.creative_weight,
        args.generic_weight,
    ]
    total_weight = sum(weights)
    assert all(w >= 0 for w in weights), "All weights must be non-negative"
    assert total_weight > 0, "At least one weight must be greater than zero"

    vllm_model = args.vllm_model
    print(f"Using model: {vllm_model}")
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=args.max_model_length,
        limit_mm_per_prompt={"image": 0},
    )

    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    shard_pattern = args.shard_pattern
    # n_qa_pairs = args.n_qa_pairs

    paths = glob(
        f"./data/raw_datasets/fineweb_edu/sample/100BT/{shard_pattern}.parquet"
    )

    split = "train[:100]" if args.debug else "train"
    for path in paths:
        ds = load_dataset(
            "parquet",
            data_files=path,
            split=split,
        )
        ds = ds.filter(
            length_filter,
            fn_kwargs={"min_len": args.min_length, "max_len": args.max_length},
            num_proc=8,
        )

        messages = []
        q_types = []
        for ctx in ds["text"]:
            message, q_type = build_messages(
                ctx,
                args.question_weight,
                args.use_case_weight,
                args.creative_weight,
                args.generic_weight,
            )
            messages.append(message)
            q_types.append(q_type)

        print(f"Generating from {len(messages)} contexts")
        completions = llm.chat(
            messages,
            sampling_params=SamplingParams(
                max_tokens=256,
                temperature=0.0,
                # needed for checking if stop tokens are present
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            ),
        )
        samples = []
        for ctx, completion, q_type in zip(ds["text"], completions, q_types):
            questions = postprocess_questions(completion.outputs[0].text)
            samples.append(
                {
                    "context": ctx,
                    "prompts": questions,
                    "type": q_type,
                }
            )
            if args.debug:
                print(f"{ctx=}")
                print(f"{completion.outputs[0].text=}")
                print(f"{q_type=}")
                for q in questions:
                    print(f"{q=}")
                    print()
                print("=" * 80)

        print(f"Generated {len(samples)} samples")
        df = pd.DataFrame(samples)
        ds = Dataset.from_pandas(df)
        val_ds = ds.take(10)
        ds = ds.skip(10)

        shard_name = path.split("/")[-1].split(".")[0]
        shard_name += "_level_0"
        if args.debug:
            shard_name += "_debug"
        ds.to_parquet(
            f"data/raw_datasets/fw_qa_v3/min_{args.min_length}_to_{args.max_length}/{shard_name}.parquet"
        )
        val_ds.to_parquet(
            f"data/raw_datasets/fw_qa_v3/min_{args.min_length}_to_{args.max_length}/{shard_name}_val.parquet"
        )
        print(
            f"Saved to data/raw_datasets/fw_qa_v3/min_{args.min_length}_to_{args.max_length}/{shard_name}.parquet"
        )
        print(
            f"Saved to data/raw_datasets/fw_qa_v3/min_{args.min_length}_to_{args.max_length}/{shard_name}_val.parquet"
        )
