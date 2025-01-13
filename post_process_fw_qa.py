import re
import json
import random
from glob import glob

import pandas as pd
import numpy as np
from datasets import Dataset


def get_repeat_prompts(ds):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    for ctx, prompt, response in zip(ds["context"], ds["prompt"], ds["response"]):
        # Only process if the context is not already in the set
        if ctx in unique_contexts:
            continue

        unique_contexts.add(ctx)
        ctxs.append(ctx)
        responses.append(ctx)
        prompts.append("Repeat the text above.")

    print(f"Adding repeat prompt...")
    print(f"# unique contexts: {len(unique_contexts)}")

    return Dataset.from_pandas(
        pd.DataFrame(
            dict(
                context=ctxs,
                response=responses,
                prompt=prompts,
            )
        )
    )


def get_negative_prompts(ds):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    keywords = [
        "repeat",
        "rephrase",
        "summarize",
        "rewrite",
        "title",
        "keyword",
        "continuation",
    ]

    for ctx, prompt, response in zip(ds["context"], ds["prompt"], ds["response"]):
        if ctx in unique_contexts:
            continue
        if any(keyword in prompt for keyword in keywords):
            # Skip samples where the prompt contains any of the specified keywords
            continue

        unique_contexts.add(ctx)
        ctxs.append(ctx)
        prompts.append(prompt)
        responses.append(response)

    print(f"Adding negative prompt...")
    print(f"# unique contexts: {len(unique_contexts)}")

    # remove one last sample if the number of samples is odd
    if len(ctxs) % 2 != 0:
        ctxs.pop()
        prompts.pop()
        responses.pop()

    # to make sure that the negative prompt/response is not the same as the original
    indices = list(np.random.permutation(len(ctxs))) + list(
        np.random.permutation(len(ctxs))
    )
    neg_ctxs, neg_prompts, neg_responses = [], [], []
    for idx in range(0, len(indices), 2):
        i = indices[idx]
        j = indices[idx + 1]
        neg_ctxs.append(ctxs[i])
        neg_prompts.append(ctxs[j] + "\n\n" + prompts[j])
        neg_responses.append(responses[j])

    return Dataset.from_pandas(
        pd.DataFrame(
            dict(
                context=neg_ctxs,
                prompt=neg_prompts,
                response=neg_responses,
            )
        )
    )


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
    if (len(questions) > 0) and (len(answers) > 0):
        for i in range(min(len(questions), len(answers))):
            out_q.append(questions[i].strip())
            out_a.append(answers[i].strip())

    return out_q, out_a


def get_response_txt(res_item):
    try:
        return res_item["response"]["body"]["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"Error getting response: {e}")
        return None


def get_prompt_txt(prompt_item):
    try:
        return prompt_item["body"]["messages"][1]["content"]
    except Exception as e:
        print(f"Error getting context: {e}")
        return None


if __name__ == "__main__":
    res_files = glob("openai_batches/fineweb_qa_pairs_*_res.jsonl")
    prompt_files = [f.replace("_res.jsonl", ".jsonl") for f in res_files]

    for res_file, prompt_file in zip(res_files, prompt_files):
        print(f"Processing {res_file} and {prompt_file}")
        idx = int(prompt_file.split("/")[-1].split("_")[-1].split(".jsonl")[0])
        res_data = []
        prompt_data = []
        samples = []

        # Load data from res_file
        with open(res_file, "r") as f:
            for line in f:
                res_data.append(json.loads(line))

        # Load data from prompt_file and create an index
        prompt_data_index = {}
        with open(prompt_file, "r") as f:
            for line in f:
                prompt_item = json.loads(line)
                prompt_data_index[prompt_item.get("custom_id")] = prompt_item

        # Match JSON objects based on custom_id using the index
        for res_item in res_data:
            custom_id = res_item.get("custom_id")
            prompt_item = prompt_data_index.get(custom_id)
            if not prompt_item:
                print(f"No prompt item found for {custom_id}")
                continue
            res_txt = get_response_txt(res_item)
            if not res_txt:
                print(f"No response text found for {custom_id}")
                continue
            questions, answers = postprocess_qa_pairs(res_txt)
            if not questions or not answers:
                print(f"No questions or answers found for {custom_id}")
                continue
            context = get_prompt_txt(prompt_item).split("### Context ###")[-1]
            for q, a in zip(questions, answers):
                samples.append(
                    {
                        "context": context,
                        "prompt": q,
                        "response": a,
                    }
                )

        print(f"Found {len(samples)} samples in {res_file} and {prompt_file}")
        random.shuffle(samples)
        df = pd.DataFrame(samples)
        ds = Dataset.from_pandas(df)
        val_ds = ds.take(100)
        ds = ds.skip(100)

        ds.to_parquet(f"data/raw_datasets/fw_qa/{idx:05d}.parquet")
        val_ds.to_parquet(f"data/raw_datasets/fw_qa/{idx:05d}_val.parquet")
        print(f"Saved to data/raw_datasets/fw_qa/{idx:05d}.parquet")
        print(f"Saved to data/raw_datasets/fw_qa/{idx:05d}_val.parquet")

        # add repeat and negative prompt
        repeat_ds = get_repeat_prompts(ds).shuffle(seed=42)
        repeat_ds.to_parquet(f"data/raw_datasets/fw_qa/{idx:05d}_repeat.parquet")
        print(f"repeat_ds: {len(repeat_ds)}")
        print(
            f"Saved {idx:05d}_repeat.parquet to data/raw_datasets/fw_qa/{idx:05d}_repeat.parquet"
        )
        neg_ds = get_negative_prompts(ds).shuffle(seed=42)
        neg_ds.to_parquet(f"data/raw_datasets/fw_qa/{idx:05d}_neg.parquet")
        print(f"neg_ds: {len(neg_ds)}")
        print(
            f"Saved {idx:05d}_neg.parquet to data/raw_datasets/fw_qa/{idx:05d}_neg.parquet"
        )
