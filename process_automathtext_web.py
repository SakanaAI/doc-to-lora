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


SYSTEM_TEMPLATE = "You are a strong math teaching assistant. You'll be reviewing a web page related to math and generating questions and answers from the page."

# based on Make Your LLM Fully Utilize the Context (https://arxiv.org/pdf/2404.16811)
PROMPT_TEMPLATE = (
    "Generate questions and corresponding answers from the given context. The questions should be highly specific to the "
    "mathematical knowledge in the context, not general questions that suits any context.\n\n"
    "Rules to follow when generate the questions:\n"
    "1. The questions must be fully answerable from mathematical information present in given context.\n"
    "2. Make sure the questions are clear and unambiguous.\n"
    "3. The questions should not focus on the formatting or LaTeX code of the context.\n"
    "4. The questions should require the mathematical knowledge and information present in the context.\n\n"
    "Rules to follow when generate the answers:\n"
    "1. The answers must use the information provided in the context.\n"
    "2. The answer should be informative and explain in detail how to arrive at the answer based on the given math content.\n"
    "3. Make sure that the explanation includes all the necessary steps and details to arrive at the answer.\n\n"
    "Response with {n_qa_pairs} question-answer pairs.\n"
    "The question-answer pairs should be in the following format:\n"
    "Question 1: {{question_1}}\n"
    "Answer 1: {{answer_1}}\n"
    "Question 2: {{question_2}}\n"
    "Answer 2: {{answer_2}}\n"
    "..."
    "\n\n"
    "### Context ###\n"
    "{txt}"
)


def get_prompt(txt, n_qa_pairs):
    prompt = PROMPT_TEMPLATE.format(txt=txt, n_qa_pairs=n_qa_pairs)
    return prompt


def get_json_request(txt, n_qa_pairs, gpt_model_name):
    prompt = get_prompt(txt, n_qa_pairs)
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
        "web-0.80-to-1.00",
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

    with open(f"openai_batches/automathtext_web_qa_pairs.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
