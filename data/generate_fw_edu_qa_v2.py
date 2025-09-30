import argparse
import os
import re
from glob import glob

import pandas as pd
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

STOP_STRINGS = {
    "google/gemma-3-12b-it": ["<eos>", "<end_of_turn>"],
}

SYSTEM_TEMPLATE = (
    "You are a creative and helpful assistant.\n"
    "You are tasked with generating questions for reading comprehension tests.\n"
    "You will be given a context and you need to generate questions and corresponding answers from the given context.\n"
    "The questions should be highly specific to the information provided in the context, not general questions that suit any context.\n"
    "**DO NOT** hallucinate or make up information."
)

# based on Make Your LLM Fully Utilize the Context (https://arxiv.org/pdf/2404.16811)
PROMPT_TEMPLATE = (
    "### Instructions ###\n"
    "Generate questions and corresponding answers from the given context. The questions should be highly specific to the "
    "information provided in the context, not general questions that suit any context.\n\n"
    "### Context ###\n"
    "{context}\n\n\n"
    "### Rules ###\n"
    "Rules to follow when generating the questions:\n"
    "1. The questions must be specific to the given context and fully answerable from information present in the given context.\n"
    "2. Ask questions that are fact-seeking based on the information provided.\n"
    "3. Make sure the questions are clear and unambiguous.\n"
    "4. Phrases like 'based on the provided context', 'according to the context', 'in the context', etc., are **NOT ALLOWED** to appear in "
    "the questions.\n"
    "5. The questions should not overlap. They should be diverse, covering many aspects of the context.\n"
    "6. Do not give away too much information in the questions. For example, ask 'Who is X?' instead of 'Who is X that did Y?' when Y is clear from the context.\n"
    "7. Ignore the text formatting of the context, e.g., bold, italic, underline, etc.\n"
    "8. Ignore typos, spacing, and grammatical errors in the context.\n\n"
    "Rules to follow when generating the answers:\n"
    "1. The answers must use the (implied) information provided in the context.\n"
    "2. Phrases like 'based on the provided context', 'according to the context', 'in the context', etc., are **NOT ALLOWED** to appear in "
    "the answers.\n"
    "3. Do not just copy words from the context. Answer the question in your own words.\n"
    "4. The answers should be detailed and comprehensive. Please include additional specific details from the context.\n\n"
    "Respond with {n_qa_pairs} question-answer pairs.\n"
    "Always use proper grammar and punctuation.\n"
    "Try to use different question forms and styles.\n"
    "Use simple words and make sure that the answers are clear and comprehensive.\n\n"
    "The question-answer pairs should be in the following format:\n"
    "Question 1: {{question_1}}\n"
    "Answer 1: {{answer_1}}\n"
    "Question 2: {{question_2}}\n"
    "Answer 2: {{answer_2}}\n"
    "..."
)


def get_prompt(context, n_qa_pairs):
    prompt = PROMPT_TEMPLATE.format(context=context, n_qa_pairs=n_qa_pairs)
    return prompt


def check_should_skip(txt: str, vllm_model: str) -> bool:
    """Check if the response should be skipped based on stop strings."""
    for stop in STOP_STRINGS[vllm_model]:
        if stop in txt[-len(stop) :]:
            return (txt.split(stop)[0], False)  # Found a valid stop string
    return (txt, True)  # No valid stop string found, skip this response


def postprocess_qa_pairs(res_txt: str):
    """
    Postprocesses the QA pairs from the response text.

    Args:
        res_txt: The response text.
        n_qa_pairs: The number of QA pairs.

    Returns:
        A tuple of two lists, the first containing the questions and the second containing the answers.
    """
    # capture everything after each "Question {number}:" until "Answer"
    res_txt = remove_think(res_txt)
    q_pattern = r"Question \d+:(.*?)(?=Answer|$)"  # thanks chatgpt
    questions = re.findall(q_pattern, res_txt, flags=re.S)

    a_pattern = r"Answer \d+:(.*?)(?=Question|$)"  # thanks chatgpt
    answers = re.findall(a_pattern, res_txt, flags=re.S)

    if len(questions) != len(answers):
        print(f"Warning---number of questions and answers do not match")
        print(f"Number of questions: {len(questions)}")
        print(f"Number of answers: {len(answers)}")

    out_q = []
    out_a = []
    n_skips = 0
    if (len(questions) > 0) and (len(answers) > 0):
        n_gen_pairs = min(len(questions), len(answers))
        has_left_over = n_gen_pairs < len(questions) or n_gen_pairs < len(answers)
        for i in range(n_gen_pairs):
            response = answers[i].strip()
            question = questions[i].strip()
            if not response or not question:
                print(f"Skipping empty question or answer at index {i}")
                continue
            if (not has_left_over) and (i == n_gen_pairs - 1):
                response, skip = check_should_skip(response, vllm_model)
                if skip:
                    print(f"Skipping due to missing stop string")
                    n_skips += 1
                    continue
            out_q.append(question.strip())
            out_a.append(response.strip())
    print(f"Skipped {n_skips} responses due to missing stop strings")

    return out_q, out_a


def length_filter(sample, min_len, max_len):
    return min_len <= len(sample["text"]) <= max_len


def remove_think(txt):
    return txt.split("</think>")[-1]


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate QA pairs from FineWeb Edu dataset"
    )
    parser.add_argument(
        "--vllm_model",
        type=str,
        default=os.environ.get("vllm_model", "google/gemma-2-27b-it"),
        help="VLLM model to use for generation",
    )
    parser.add_argument(
        "--shard_pattern",
        type=str,
        required=True,
        help="Pattern to match shard files (e.g., '000_0000*')",
    )
    parser.add_argument(
        "--n_qa_pairs",
        type=int,
        required=True,
        help="Number of question-answer pairs to generate per context",
    )
    parser.add_argument(
        "--min_length",
        type=int,
        default=0,
        help="Minimum length of the context to consider for generation",
    )
    parser.add_argument(
        "--max_length",
        type=int,
        default=2000,
        help="Maximum length of the context to consider for generation",
    )
    parser.add_argument(
        "--max_model_length",
        type=int,
        default=2**14,
        help="Maximum length of the model input (context + prompt + response) in tokens",
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Debug mode - process only first 100 samples",
    )

    args = parser.parse_args()
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
    n_qa_pairs = args.n_qa_pairs

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

        ctxs = [sample["text"] for sample in iter(ds)]
        messages = [
            [
                {"role": "system", "content": SYSTEM_TEMPLATE},
                {"role": "user", "content": get_prompt(ctx, n_qa_pairs)},
            ]
            for ctx in ctxs
        ]

        print(f"Generating from {len(messages)} contexts")
        completions = llm.chat(
            messages,
            sampling_params=SamplingParams(
                max_tokens=2048,
                temperature=0.0,
                # needed for checking if stop tokens are present
                skip_special_tokens=False,
                include_stop_str_in_output=True,
            ),
        )
        samples = []
        for ctx, completion in zip(ctxs, completions):
            questions, answers = postprocess_qa_pairs(completion.outputs[0].text)
            samples.append(
                {
                    "context": ctx,
                    "prompts_level_0": questions,
                    "responses_level_0": answers,
                }
            )
            if args.debug:
                print(f"{ctx=}")
                print(f"{completion.outputs[0].text=}")
                for q, a in zip(questions, answers):
                    print(f"{q=}")
                    print(f"{a=}")
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
            f"data/raw_datasets/fw_qa_v2/min_{args.min_length}_to_{args.max_length}/{shard_name}.parquet"
        )
        val_ds.to_parquet(
            f"data/raw_datasets/fw_qa_v2/min_{args.min_length}_to_{args.max_length}/{shard_name}_val.parquet"
        )
        print(
            f"Saved to data/raw_datasets/fw_qa_v2/min_{args.min_length}_to_{args.max_length}/{shard_name}.parquet"
        )
        print(
            f"Saved to data/raw_datasets/fw_qa_v2/min_{args.min_length}_to_{args.max_length}/{shard_name}_val.parquet"
        )
