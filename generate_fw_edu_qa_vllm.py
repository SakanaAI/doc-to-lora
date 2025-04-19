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
    "You are tasked with generating questions for reading comprehension tests.\n"
    "You will be given a context and you need to generate questions and corresponding answers from the given context.\n"
    "The questions should be highly specific to the information provided in the context, not general questions that suits any context.\n"
    "**DO NOT** halucinate or make up information."
)

# based on Make Your LLM Fully Utilize the Context (https://arxiv.org/pdf/2404.16811)
PROMPT_TEMPLATE = (
    "### Instructions ###\n"
    "Generate questions and corresponding answers from the given context. The questions should be highly specific to the "
    "information provided in the context, not general questions that suits any context.\n\n\n"
    "### Context ###\n"
    "{context}"
    "\n\n\n"
    "### Additional Instructions ###\n"
    "Rules to follow when generate the questions:\n"
    "1. The questions must be specific to the given context and fully answerable from information present in given context.\n"
    "2. Make sure the questions are clear and unambiguous.\n"
    "3. Phrases like 'based on the provided context', 'according to the context', etc, are **NOT ALLOWED** to appear in "
    "the questions.\n"
    "4. The questions should not overlap. They should be diverse, covering many aspects of the context.\n"
    "5. Do not give away too much information in the questions. For example, ask 'Who is X' instead of 'Who is X that did Y' when Y is clear from the context.\n"
    "6. Ignore the text formatting of the context, e.g., bold, italic, underline, etc.\n"
    "7. Ignore typos, spacing and grammatical errors in the context.\n\n"
    "Rules to follow when generate the answers:\n"
    "1. The answers must use the information provided in the context.\n"
    "2. Do not just copy words from the context. Answer the question in your own words.\n"
    "3. The answers should be detailed and comprehensive. Please include additional specific details from the context when possible.\n\n"
    "Response with {n_qa_pairs} question-answer pairs.\n"
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


def length_filter(samples):
    return [len(text) < 10_000 for text in samples["text"]]


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
    mm_kwargs = {"image":0} # if "gemma-3" in vllm_model else {}
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        # tokenizer_mode="mistral",
        # config_format="mistral",
        # load_format="mistral",
        enable_prefix_caching=True,
        max_model_len=11264,#2**14,
        limit_mm_per_prompt=mm_kwargs,
        # enable_chunked_prefill=True,
    )
    
    if vllm_model=="mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
        SYSTEM_TEMPLATE = (
            "You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\n"
            "You power an AI assistant called Le Chat.\n"
            "Your knowledge base was last updated on 2023-10-01.\n\n\n"
        ) + SYSTEM_TEMPLATE
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    shard_pattern = sys.argv[1]
    n_qa_pairs = int(sys.argv[2])

    paths = glob(f"./data/raw_datasets/fineweb_edu/sample/100BT/{shard_pattern}.parquet")

    for path in paths:
        ds = load_dataset(
            "parquet",
            data_files=path,
            split="train",
            streaming=True,
        )
        ds = ds.filter(length_filter, batched=True)

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
                temperature=0.1 if vllm_model=="mistralai/Mistral-Small-3.1-24B-Instruct-2503" else 0.7,
                # frequency_penalty=0.1,
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
        ds.to_parquet(f"data/raw_datasets/fw_qa_3/{shard_name}.parquet")
        val_ds.to_parquet(f"data/raw_datasets/fw_qa_3/{shard_name}_val.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_3/{shard_name}.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_3/{shard_name}_val.parquet")

    # debug
    # ds = load_dataset(
    #     "parquet",
    #     data_files=paths[0],
    #     split="train[:10]",
    # )
    # ctxs = [sample["text"] for sample in iter(ds)]
    # messages = [
    #     [
    #         {"role": "system", "content": SYSTEM_TEMPLATE},
    #         {"role": "user", "content": get_prompt(ctx, n_qa_pairs)},
    #     ]
    #     for ctx in ctxs
    # ]
    #
    # print(f"Generating from {len(messages)} contexts")
    # completions = llm.chat(
    #     messages,
    #     sampling_params=SamplingParams(
    #         max_tokens=2048,
    #         temperature=0.1 if vllm_model=="mistralai/Mistral-Small-3.1-24B-Instruct-2503" else 0.7,
    #         # frequency_penalty=0.1,
    #     ),
    # )
    # samples = []
    # for ctx, completion in zip(ctxs, completions):
    #     questions, answers = postprocess_qa_pairs(completion.outputs[0].text)
    #     for q, a in zip(questions, answers):
    #         samples.append(
    #             {
    #                 "context": ctx,
    #                 "prompt": q,
    #                 "response": a,
    #             }
    #         )
    # for sample in samples:
    #     print(f"{sample['context']=}")
    #     print(f"{sample['prompt']=}")
    #     print(f"{sample['response']=}")
    #     print()
