import random
import os
import json
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
    "Generate questions and corresponding answers from the given context. The questions should be highly specific to the "
    "information provided in the context, not general questions that suits any context.\n"
    "Finally, provide a plausible 1-paragraph continuation of the context. "
    "The continuation should reference information in the context in some way.\n\n"
    "Rules to follow when generate the questions:\n"
    "1. The questions must be fully answerable from information present in given context.\n"
    "2. Make sure the questions are clear and unambiguous.\n"
    "3. Phrases like 'based on the provided context', 'according to the context', etc, are not allowed to appear in "
    "the questions.\n\n"
    "Rules to follow when generate the answers:\n"
    "1. The answers must use the information provided in the context.\n"
    "2. Do not just copy words from the context. Answer the question in your own words.\n\n"
    "Response with {n_qa_pairs} question-answer pairs.\n"
    "After the question-answer pairs, provide a plausible continuation of the context.\n"
    "Use simple words and please be clear.\n"
    "The question-answer pairs should be in the following format:\n"
    "Question 1: {{question_1}}\n"
    "Answer 1: {{answer_1}}\n"
    "Question 2: {{question_2}}\n"
    "Answer 2: {{answer_2}}\n"
    "...\n"
    "Continuation: {{continuation}}\n"
    "\n\n"
    "### Context ###\n"
    "{context}"
)


def get_prompt(context, n_qa_pairs):
    prompt = PROMPT_TEMPLATE.format(context=context, n_qa_pairs=n_qa_pairs)
    return prompt


def get_json_request(id, text, n_qa_pairs, gpt_model_name):
    prompt = get_prompt(text, n_qa_pairs)
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
    return [len(text) < 10_000 for text in samples["markdown"]]


if __name__ == "__main__":
    set_seed(42)
    ds = load_dataset("open-phi/programming_books_llama", split="train")
    ds = ds.filter(remove_too_long, batched=True)
    print(f"Filtered ds size: {len(ds)}")

    os.makedirs("openai_batches", exist_ok=True)
    lines = []
    for i, sample in tqdm(enumerate(ds)):
        jsonl = get_json_request(
            f"openphi_prog_{i}",
            sample["markdown"],
            n_qa_pairs=2,
            gpt_model_name="gpt-4o-mini",
        )
        lines.append(jsonl)

    with open(f"openai_batches/openphi_prog_qa_pairs.jsonl", "w") as f:
        for line in lines:
            f.write(json.dumps(line) + "\n")
