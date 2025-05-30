import os
import random
import sys
from glob import glob

import pandas as pd
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

SYSTEM_TEMPLATE = (
    "### System Instructions ###\n"
    "You are a creative and helpful assistant.\n"
    "**DO NOT** halucinate or make up information."
)

PROMPT_TEMPLATE = "### Context ###\n{context}\n\n\n### Question ###\n{question}"

MODEL_CTX_LEN = {
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
}


def get_prompt(context, q):
    prompt = PROMPT_TEMPLATE.format(context=context, question=q)
    return prompt


if __name__ == "__main__":
    vllm_model = os.environ.get("vllm_model")  # "google/gemma-2-27b-it"
    debug = bool(os.environ.get("debug", False))
    print(f"Using model: {vllm_model}")
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=MODEL_CTX_LEN.get(vllm_model),
    )

    if vllm_model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        mm_kwargs = {"image": 0}  # if "gemma-3" in vllm_model else {}
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
        SYSTEM_TEMPLATE = (
            "You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\n"
            "You power an AI assistant called Le Chat.\n"
            "Your knowledge base was last updated on 2023-10-01.\n\n\n"
        ) + SYSTEM_TEMPLATE
        llm_kwargs["max_model_len"] = 2**14
        llm_kwargs["limit_mm_per_prompt"] = mm_kwargs

    print(f"{llm_kwargs=}")
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    shard_pattern = sys.argv[1]

    paths = glob(f"./data/raw_datasets/fw_qa_3/{shard_pattern}.parquet")

    for path in paths:
        if debug:
            ds = load_dataset(
                "parquet",
                data_files=path,
                split="train[:10]",
            )
        else:
            ds = load_dataset(
                "parquet",
                data_files=path,
                split="train",
                # streaming=True,
            )
        ctxs = []
        questions = []
        for sample in iter(ds):
            ctxs.append(sample["context"])
            questions.append(sample["prompt"])
        if "gemma" in vllm_model:
            # gemma models do not support system messages
            messages = [
                [
                    {
                        "role": "user",
                        "content": SYSTEM_TEMPLATE + "\n\n\n" + get_prompt(ctx, q),
                    },
                ]
                for ctx, q in zip(ctxs, questions)
            ]
        else:
            messages = [
                [
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": get_prompt(ctx, q)},
                ]
                for ctx, q in zip(ctxs, questions)
            ]

        print(f"Generating from {len(messages)} contexts")
        completions = llm.chat(
            messages,
            sampling_params=SamplingParams(
                max_tokens=2048,
                temperature=(
                    0.1
                    if vllm_model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                    else 1.0
                ),
            ),
        )
        samples = []
        for ctx, q, completion in zip(ctxs, questions, completions):
            response = completion.outputs[0].text
            samples.append(
                {
                    "context": ctx,
                    "prompt": q,
                    "response": response,
                }
            )
            if debug:
                print(f"{ctx=}")
                print(f"{q=}")
                print(f"{response=}")
                print("=" * 80)

        print(f"Generated {len(samples)} samples")
        random.shuffle(samples)
        df = pd.DataFrame(samples)
        ds = Dataset.from_pandas(df)

        shard_name = path.split("/")[-1].split(".")[0]
        ds.to_parquet(f"data/raw_datasets/fw_qa_3/{vllm_model}/{shard_name}.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_3/{vllm_model}/{shard_name}.parquet")
