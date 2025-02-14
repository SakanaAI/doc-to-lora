from glob import glob
import os
import re
import time
import random
import json
from numpy import mat
import requests
from dataclasses import dataclass, asdict
from typing import List, Optional

import yaml
from openai import OpenAI
from datasets import load_dataset
from argparse import ArgumentParser, Namespace
from datetime import datetime
from tqdm import tqdm


from hyper_llm_modulator.utils import get_preprocessing_fn


# NOTE: Please, store your openai key with "export OPENAI_API_KEY=..."
api_key = os.environ.get("OPENAI_API_KEY")


SYSTEM_TEMPLATE = "You are a creative and helpful assistant."

# based on Make Your LLM Fully Utilize the Context (https://arxiv.org/pdf/2404.16811)
PROMPT_TEMPLATE = (
    "Generate questions and corresponding answers from the given context. The questions should be highly specific to the "
    "information provided in the context, not general questions that suits any context.\n\n"
    "Rules to follow when generate the questions:\n"
    "1. The questions must be fully answerable from information present in given context.\n"
    "2. Make sure the questions are clear and unambiguous.\n"
    "3. Phrases like 'based on the provided context', 'according to the context', etc, are not allowed to appear in "
    "the questions.\n\n"
    "Rules to follow when generate the answers:\n"
    "1. The answers must use the information provided in the context.\n"
    "2. Do not just copy words from the context. Answer the question in your own words.\n\n"
    "Response with {n_qa_pairs} question-answer pairs. Use simple words and please be clear.\n"
    "The question-answer pairs should be in the following format:\n"
    "Question 1: {{question_1}}\n"
    "Answer 1: {{answer_1}}\n"
    "Question 2: {{question_2}}\n"
    "Answer 2: {{answer_2}}\n"
    "..."
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
        "custom_id": f"{id}",
        "method": "POST",
        "url": "/v1/chat/completions",
        "body": {
            "model": gpt_model_name,
            "messages": messages,
            "temperature": 1.0,
            "frequency_penalty": 0.2,
        },
    }


def parse_args() -> Namespace:
    p = ArgumentParser()
    p.add_argument("--data_dir", type=str, default="tasks")

    # Every second, different unique id
    current_time = datetime.now().strftime("%Y%m%d%H%M%S")
    default_output_dir = f"generated_tasks/{current_time}"
    p.add_argument("--generated_data_dir", type=str, default=default_output_dir)
    p.add_argument("--gpt_model_name", type=str, default="gpt-4o-mini")
    p.add_argument("--n_qa_pairs", type=int, default=2)
    p.add_argument("--shard_pattern", type=str, default="*")

    return p.parse_args()


if __name__ == "__main__":
    # Load the datasets
    args = parse_args()
    ds = load_dataset(
        "parquet",
        data_files=f"./data/raw_datasets/fineweb_sharded/{args.shard_pattern}.parquet",
        split="train",
        streaming=True,
    )
    os.makedirs("openai_batches", exist_ok=True)
    lines = []
    c = 0
    for i, sample in tqdm(enumerate(iter(ds))):
        if len(lines) >= 20_000:
            with open(f"openai_batches/fineweb_qa_pairs_{c}.jsonl", "w") as f:
                for line in lines:
                    f.write(json.dumps(line) + "\n")
            lines = []
            c += 1
        jsonl = get_json_request(
            f"{c}_{i}", sample["text"], args.n_qa_pairs, args.gpt_model_name
        )
        lines.append(jsonl)

    client = OpenAI()
    res_files = glob("openai_batches/fineweb_qa_pairs_*_res.jsonl")
    prompt_files = glob("openai_batches/fineweb_qa_pairs_*[!res].jsonl")
    unprocessed_files = set(prompt_files) - set(
        [f.replace("_res", "") for f in res_files]
    )
    for file in unprocessed_files:
        print(f"Submitting batch {file}")
        batch_input_file = client.files.create(file=open(file, "rb"), purpose="batch")
        batch_input_file_id = batch_input_file.id
        batch_info = client.batches.create(
            input_file_id=batch_input_file_id,
            endpoint="/v1/chat/completions",
            completion_window="24h",
            metadata={"description": f"{file}"},
        )
        print(f"Batch {file} submitted")
        print(batch_info)
        print("-" * 100)

        while (
            batch_info.completed_at is None
            and batch_info.failed_at is None
            and batch_info.expired_at is None
            and batch_info.cancelled_at is None
        ):
            time.sleep(10)
            batch_info = client.batches.retrieve(batch_info.id)
            print(f"curtime: {datetime.now()}")
            print(batch_info)
            print()

        if batch_info.status == "completed":
            print(f"Batch {file} completed")
            res = client.files.content(batch_info.output_file_id)

            with open(f"{file.replace('.jsonl', '_res.jsonl')}", "w") as f:
                for line in res.iter_lines():
                    json_line = json.loads(line)
                    if json_line["error"]:
                        print(f"Error: {json_line['error']}")
                        print("Skipping line...")
                        continue
                    f.write(json.dumps(json_line) + "\n")
        else:
            print(f"Batch {file} failed")
            print(batch_info)
            print("-" * 100)

        time.sleep(10)
