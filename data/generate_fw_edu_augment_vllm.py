import os
import random
import sys
from glob import glob

import pandas as pd
from datasets import Dataset, load_dataset
from vllm import LLM, SamplingParams

from ctx_to_lora.utils import clear_gpu

MODEL_CTX_LEN = {
    "google/gemma-2-27b-it": 8192,
    "google/gemma-2-2b-it": 8192,
    "google/gemma-2-9b-it": 8192,
    # not actually 16k but it's faster
    "mistralai/Mistral-Small-3.1-24B-Instruct-2503": 2**14,  # 11264,
}

# based on New News: system-2 finetuning (https://arxiv.org/pdf/2505.01812)
SYSTEM_TEMPLATE = (
    "You are an careful and creative paraphraser. Paraphrase the given content without leaving out any important information. "
    "You will be given a context and you must generate a new variation of the provided context.\n"
    "You only output the paraphrase itself.\n"
    "**DO NOT** halucinate or make up information."
)

# based on On the generalization of language models from in-context learning and finetuning: a controlled study
# https://arxiv.org/pdf/2505.00661
PROMPT_TEMPLATE = (
    "### Context ###\n"
    "{context}"
    "\n\n\n"
    "### Instructions ###\n"
    "- Please generate rephrasings that can be inferred from each sentence. Even a simple sentence has some alternative phrasings that are logically equivalent.\n"
    "- Reversing a sentence is also valid but make sure that the reversed sentence is still logically equivalent.\n"
    "- You can shuffle the order of sentences as long as the content is coherent and the meaning is preserved.\n"
    "- You can combine multiple sentences into one, or split a long sentence into multiple shorter ones.\n"
    "- You can combine reversal and paraphasing, e.g., reversing a sentence and then paraphrasing it.\n"
    "- You can also add new sentences that can be logically inferred from the context. "
    "Please only use logic and language to draw your conclusions, regardless of whether the entities in question actually exist.\n"
    "- Try to keep the length to be similar to the original context\n"
    "- Please accurately maintain facts, such as names and numbers, mentioned in the context."
    "\n\nOnly output a new variation of the context. Don't include any other words."
)


def get_prompt(context):
    prompt = PROMPT_TEMPLATE.format(context=context)
    return prompt


def length_filter(samples):
    return [len(text) < 10_000 for text in samples["text"]]


if __name__ == "__main__":
    vllm_model = os.environ.get("vllm_model")
    print(f"Using model: {vllm_model}")
    llm_kwargs = dict(
        model=vllm_model,
        dtype="bfloat16",
        enable_prefix_caching=True,
        enable_chunked_prefill=True,
        max_model_len=MODEL_CTX_LEN[vllm_model],
    )

    if vllm_model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503":
        mm_kwargs = {"image": 0}
        llm_kwargs["tokenizer_mode"] = "mistral"
        llm_kwargs["config_format"] = "mistral"
        llm_kwargs["load_format"] = "mistral"
        llm_kwargs["limit_mm_per_prompt"] = mm_kwargs
        SYSTEM_TEMPLATE = (
            "You are Mistral Small 3.1, a Large Language Model (LLM) created by Mistral AI, a French startup headquartered in Paris.\n"
            "You power an AI assistant called Le Chat.\n"
            "Your knowledge base was last updated on 2023-10-01.\n\n\n"
        ) + SYSTEM_TEMPLATE
    llm = LLM(**llm_kwargs)
    tokenizer = llm.get_tokenizer()
    shard_pattern = sys.argv[1]

    paths = glob(
        f"./data/raw_datasets/fineweb_edu/sample/100BT/{shard_pattern}.parquet"
    )

    for path in paths:
        ds = load_dataset(
            "parquet",
            data_files=path,
            split="train",
            # streaming=True,
        )
        ds = ds.filter(length_filter, batched=True)

        ctxs = [sample["text"] for sample in iter(ds)]
        if "gemma" in vllm_model:
            messages = [
                [
                    {
                        "role": "user",
                        "content": SYSTEM_TEMPLATE + "\n\n\n" + get_prompt(ctx),
                    },
                ]
                for ctx in ctxs
            ]
        else:
            messages = [
                [
                    {"role": "system", "content": SYSTEM_TEMPLATE},
                    {"role": "user", "content": get_prompt(ctx)},
                ]
                for ctx in ctxs
            ]

        print(f"Generating from {len(messages)} contexts")
        completions = llm.chat(
            messages,
            sampling_params=SamplingParams(
                max_tokens=2048,
                temperature=(
                    0.2
                    if vllm_model == "mistralai/Mistral-Small-3.1-24B-Instruct-2503"
                    else 0.7
                ),
                # frequency_penalty=0.1,
            ),
        )
        samples = []
        for ctx, completion in zip(ctxs, completions):
            samples.append(
                {
                    "context": ctx,
                    "variation": completion.outputs[0].text.strip(),
                }
            )
        for sample in samples:
            print(f"{sample['context']=}")
            print(f"{sample['variation']=}")
            print("=" * 60)

        print(f"Generated {len(samples)} samples")
        random.shuffle(samples)
        df = pd.DataFrame(samples)
        ds = Dataset.from_pandas(df)
        val_ds = ds.take(10)
        ds = ds.skip(10)

        shard_name = path.split("/")[-1].split(".")[0]
        ds.to_parquet(f"data/raw_datasets/fw_qa_aug_pretrain/{shard_name}.parquet")
        val_ds.to_parquet(
            f"data/raw_datasets/fw_qa_aug_pretrain/{shard_name}_val.parquet"
        )
        print(f"Saved to data/raw_datasets/fw_qa_aug_pretrain/{shard_name}.parquet")
        print(f"Saved to data/raw_datasets/fw_qa_aug_pretrain/{shard_name}_val.parquet")
        del ds, val_ds, df, samples, completions, messages, ctxs
        clear_gpu()
