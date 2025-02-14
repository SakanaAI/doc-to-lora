import os
import sys
import random
import re
from glob import glob

import pandas as pd
from datasets import Dataset
from vllm import LLM, SamplingParams
from tqdm import tqdm
from datasets import load_dataset
from openai import OpenAI

SYSTEM_TEMPLATE = (
    "You are a creative and helpful assistant.\n"
    "You are given a context and you need to generate questions and corresponding answers from the given context.\n"
    "The questions should be highly specific to the information provided in the context, not general questions that suits any context.\n"
    "Do not halucinate and make up information."
)

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
    "2. Do not just copy words from the context. Answer the question in your own words.\n"
    "3. The answers should be detailed and comprehensive.\n\n"
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


if __name__ == "__main__":
    # api_key = os.environ.get("vllm_api_key")
    # vllm_port = os.environ.get("vllm_port")
    # vllm_ip = os.environ.get("vllm_ip")  # "172.16.0.62"
    vllm_model = os.environ.get("vllm_model")  # "google/gemma-2-27b-it"
    print(f"Using model: {vllm_model}")
    # print(f"Using API key: {api_key}")
    # print(f"Using VLLM IP: {vllm_ip}")
    # print(f"Using VLLM port: {vllm_port}")
    # client = OpenAI(
    #     base_url=f"http://{vllm_ip}:{vllm_port}/v1",
    #     api_key=api_key,
    # )
    llm = LLM(
        model=vllm_model,
        tokenizer_mode="mistral",
        config_format="mistral",
        load_format="mistral",
        enable_prefix_caching=True,
        # enable_chunked_prefill=True,
    )
    tokenizer = llm.get_tokenizer()
    shard_pattern = sys.argv[1]
    n_qa_pairs = int(sys.argv[2])

    for path in glob(f"./data/raw_datasets/fineweb_sharded/{shard_pattern}.parquet"):
        ds = load_dataset(
            "parquet",
            data_files=path,
            split="train",
            streaming=True,
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
                temperature=1.0,
                frequency_penalty=0.2,
            ),
        )
        samples = []
        for ctx, completion in zip(ctxs, completions):
            questions, answers = postprocess_qa_pairs(completion.outputs[0].text)
            for q, a in zip(questions, answers):
                samples.append(
                    {
                        "context": ctx,
                        "prompt": q,
                        "response": a,
                    }
                )

        print(f"Generated {len(samples)} samples")
        random.shuffle(samples)
        df = pd.DataFrame(samples)
        ds = Dataset.from_pandas(df)
        val_ds = ds.take(10)
        ds = ds.skip(10)

        shard_name = path.split("/")[-1].split(".")[0]
        ds.to_parquet(f"data/raw_datasets/fw_qa_xl/{shard_name}.parquet")
        val_ds.to_parquet(f"data/raw_datasets/fw_qa_xl/{shard_name}_val.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_xl/{shard_name}.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_xl/{shard_name}_val.parquet")

    # for i, sample in tqdm(enumerate(iter(ds))):
    #     completion = client.chat.completions.create(
    #         model=vllm_model,
    #         messages=[
    #             {"role": "system", "content": SYSTEM_TEMPLATE},
    #             {"role": "user", "content": get_prompt(sample["text"], n_qa_pairs)},
    #         ],
    #         extra_body={"temperature": 1.0, "frequency_penalty": 0.2},
    #     )

    #     print(completion.choices[0].message)
    #     if i >= 10:
    #         break
