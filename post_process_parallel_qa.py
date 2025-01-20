import json
import re
import os
from glob import glob

import numpy as np
import pandas as pd
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

    c_pattern = r"Continuation:\s*(.*)"  # thanks chatgpt
    continuations = re.findall(c_pattern, res_txt, flags=re.S)

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

    if len(continuations) > 0:
        assert len(continuations) == 1
        continuations[0] = continuations[0].strip()

    return out_q, out_a, continuations


if __name__ == "__main__":
    file_pattern = os.sys.argv[1]
    files = glob(file_pattern)
    print(f"Files: {files}")
    print(f"Processing {len(files)} files")
    for file in files:
        fname = os.path.basename(file).replace(".jsonl", "")
        samples = []
        unique_contexts = set()
        with open(file, "r") as f:
            while True:
                line = f.readline()
                if not line:
                    break  # End of file
                try:
                    json_obj = json.loads(line)
                except Exception as e:
                    print(f"Error loading JSON: {e}")
                    continue

                if not isinstance(json_obj[1], dict):
                    print(f"Error: {json_obj[1]} is not a dictionary")
                    continue

                prompt_txt = json_obj[0]["messages"][1]["content"].split(
                    "### Context ###"
                )[-1]
                res_txt = json_obj[1]["choices"][0]["message"]["content"]
                questions, answers, continuations = postprocess_qa_pairs(res_txt)
                unique_contexts.add(prompt_txt)
                for q, a in zip(questions, answers):
                    samples.append(
                        {
                            "context": prompt_txt,
                            "prompt": q,
                            "response": a,
                        }
                    )
                if continuations:
                    samples.append(
                        {
                            "context": prompt_txt,
                            "prompt": "Write a 1-paragraph continuation of the text.",
                            "response": continuations[0],
                        }
                    )
        df = pd.DataFrame(samples)

        # raise NotImplementedError
        ds = Dataset.from_pandas(df).shuffle(seed=42)
        print(
            f"Created {len(ds)} samples from {file} with {len(unique_contexts)} unique contexts"
        )
        val_ds = ds.take(500)
        ds = ds.skip(500)
        ds.to_parquet(f"data/raw_datasets/ctx_qa/{fname}.parquet")
        val_ds.to_parquet(f"data/raw_datasets/ctx_qa/{fname}_val.parquet")
        print(f"Saved {fname} to data/raw_datasets/ctx_qa/{fname}.parquet")
        print(f"Saved {fname}_val to data/raw_datasets/ctx_qa/{fname}_val.parquet")
        # add repeat and negative prompt
        repeat_ds = get_repeat_prompts(ds).shuffle(seed=42)
        repeat_ds.to_parquet(f"data/raw_datasets/ctx_qa/{fname}_repeat.parquet")
        print(f"repeat_ds: {len(repeat_ds)}")
        print(
            f"Saved {fname}_repeat.parquet to data/raw_datasets/ctx_qa/{fname}_repeat.parquet"
        )
        neg_ds = get_negative_prompts(ds).shuffle(seed=42)
        neg_ds.to_parquet(f"data/raw_datasets/ctx_qa/{fname}_neg.parquet")
        print(f"neg_ds: {len(neg_ds)}")
        print(
            f"Saved {fname}_neg.parquet to data/raw_datasets/ctx_qa/{fname}_neg.parquet"
        )
        # val_ds = ds.take(500)
        # ds = ds.skip(500)
        # ds.to_parquet(f"data/raw_datasets/fw_qa_large/{fname}.parquet")
        # val_ds.to_parquet(f"data/raw_datasets/fw_qa_large/{fname}_val.parquet")
        # print(f"Saved {fname} to data/raw_datasets/fw_qa_large/{fname}.parquet")
        # print(f"Saved {fname}_val to data/raw_datasets/fw_qa_large/{fname}_val.parquet")
