import random
import os
import json
import ast

from glob import glob
from tqdm import tqdm
import numpy as np
import matplotlib.pyplot as plt
from datasets import load_dataset, Dataset
from huggingface_hub import snapshot_download
from transformers import set_seed


# NOTE: Please, store your openai key with "export OPENAI_API_KEY=..."
api_key = os.environ.get("OPENAI_API_KEY")


SYSTEM_TEMPLATE = "You are a creative and helpful assistant."

# based on Make Your LLM Fully Utilize the Context (https://arxiv.org/pdf/2404.16811)
PROMPT_TEMPLATE = (
    "Generate questions and corresponding answers from the given code. The questions should be highly specific to the "
    "algorithm implemented in the code, not general questions that suits any code.\n\n"
    "Rules to follow when generate the questions:\n"
    "1. The questions must be fully answerable from information present in given code. "
    "For example, 'What does function `add_two_numbers` do?' or 'What is the purpose of variable `x` in function `add_two_numbers`?'\n"
    "2. Make sure the questions are clear and unambiguous.\n\n"
    "Rules to follow when generate the answers:\n"
    "1. The answers must use the information provided in the code. "
    "For example, 'It adds two numbers by using the `+` operator.' or "
    "'Variable `x` in function `add_two_numbers` is used to store the result of the addition.'\n\n"
    "Response with {n_qa_pairs} question-answer pairs.\n"
    "Be creative and think of questions that are not obvious.\n"
    "Use simple words and please be clear.\n"
    "The question-answer pairs should be in the following format:\n"
    "Question 1: {{question_1}}\n"
    "Answer 1: {{answer_1}}\n"
    "Question 2: {{question_2}}\n"
    "Answer 2: {{answer_2}}\n"
    "..."
    "\n\n"
    "### Code ###\n"
    "{code}"
)


def get_prompt(code, n_qa_pairs):
    prompt = PROMPT_TEMPLATE.format(code=code, n_qa_pairs=n_qa_pairs)
    return prompt


def get_json_request(code, n_qa_pairs, gpt_model_name):
    prompt = get_prompt(code, n_qa_pairs)
    messages = [
        {"role": "system", "content": SYSTEM_TEMPLATE},
        {"role": "user", "content": prompt},
    ]
    return {
        "model": gpt_model_name,
        "messages": messages,
        "temperature": 1.0,
        "frequency_penalty": 0.2,
    }


def remove_too_long(samples):
    return [len(text) < 10_000 for text in samples["text"]]


if __name__ == "__main__":
    set_seed(42)
    ds = load_dataset(
        "math-ai/AutoMathText",
        "code-python-0.80-to-1.00",
        split="train",
        trust_remote_code=True,
    )
    ds = ds.filter(remove_too_long, batched=True)
    print(f"Filtered ds size: {len(ds)}")

    os.makedirs("openai_batches", exist_ok=True)
    lines = []
    for sample in tqdm(ds):
        code = sample["text"]
        if not code:
            continue
        jsonl = get_json_request(code, n_qa_pairs=2, gpt_model_name="gpt-4o-mini")
        lines.append(jsonl)

    with open(f"openai_batches/automathtext_code_python_qa_pairs.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
