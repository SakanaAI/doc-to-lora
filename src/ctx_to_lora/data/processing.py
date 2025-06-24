import hashlib
import json
import logging
import os
import random
from collections.abc import Callable
from glob import glob
from os import path
from typing import Any

import datasets
from datasets import Dataset, is_caching_enabled, load_dataset
from transformers import PreTrainedTokenizerBase

from ctx_to_lora.data.definitions import (
    CLOSED_QA_INTX_TEMPLATES,
    DS_KWARGS,
    EVAL_INTX_TEMPLATES,
    IGNORE_INDEX,
    REPEAT_PROMPTS,
    SELF_GEN_DATA_DIR,
    TRANSFORMED_DATA_DIR,
)
from ctx_to_lora.data.packing import pack_batch
from ctx_to_lora.utils import TRAINING_TASK, concat_list

logger = logging.getLogger()


def load_answers(ds_name, split):
    if ds_name.startswith("longbench"):

        def extract_ans(sample):
            return {"answers": sample["answers"]}

    elif ds_name == "squad":

        def extract_ans(sample):
            return {"answers": sample["answers"]["text"]}

    ds_kwargs = get_ds_kwargs(ds_name, split)
    ds = load_dataset(**ds_kwargs, trust_remote_code=True)
    ds = ds.map(extract_ans, num_proc=8, remove_columns=ds.column_names)
    return ds


def closed_qa_prompting(prompt: str):
    template = random.choice(CLOSED_QA_INTX_TEMPLATES)
    return template.format(input=prompt)


def get_repeat_prompt():
    return random.choice(REPEAT_PROMPTS)


def get_preprocessing_fn(
    ds_name: str,
    is_eval: bool,
    is_pretrain: bool,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get preprocessing function for a specific dataset.

    Args:
        ds_name: Name of the dataset

    Returns:
        A preprocessing function that takes and returns a dictionary
    """
    f = lambda x: x

    if "fw_qa_v2" in ds_name and "level" in ds_name:

        def f(sample):
            # get questions/answers from all levels in the ds
            q_cols = [col for col in sample.keys() if col.startswith("prompts_level")]
            r_cols = [col for col in sample.keys() if col.startswith("responses_level")]
            questions = concat_list([sample[col] for col in q_cols])
            responses = concat_list([sample[col] for col in r_cols])
            min_len = min(len(questions), len(responses))

            if min_len == 0:
                return {
                    "context": None,
                    "prompts": None,
                    "responses": None,
                }

            return {
                "context": sample["context"],
                "prompts": questions[:min_len],
                "responses": responses[:min_len],
            }

    elif ds_name.startswith("longbench"):

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["input"],
                "response": sample["answers"][0],
            }

    elif ds_name == "negative_nq":

        def f(sample):
            q = sample["prompt"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["context"],
                "prompt": prompt,
                "response": sample["answer"],
            }

    elif ds_name == "triviaqa_retrieved":
        # only used for eval
        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name.startswith("pwc"):

        def f(sample):
            return {
                "context": sample["input"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    elif ds_name.startswith("hotpot_qa"):

        def f(sample):
            txt = ""
            for p in sample["context"]["sentences"]:
                txt += " " + "".join(p)

            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q

            return {
                "context": txt.strip(),
                "prompt": prompt,
                "response": sample["answer"],
            }

    elif ds_name.startswith("squad"):

        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["context"],
                "prompt": prompt,
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "drop":

        def f(sample):
            q = sample["question"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["passage"],
                "prompt": prompt,
                "response": ", ".join(set(sample["answers_spans"]["spans"])),
            }

    elif ds_name in ["narrativeqa", "quoref", "ropes"]:

        def f(sample):
            response = sample["answers"][0]
            if isinstance(response, list):
                response = response[0]
            q = sample["messages"][-1]["content"]
            prompt = closed_qa_prompting(q) if not is_eval else q
            return {
                "context": sample["document"],
                "prompt": prompt,
                "response": response,
            }

    elif ds_name == "synthetic_convqa":

        def f(sample):
            response = sample["answers"][0]
            if isinstance(response, list):
                response = response[0]
            return {
                "context": sample["document"],
                "prompt": sample["messages"][-1]["content"],
                "response": response,
            }

    elif ds_name == "booksum":
        prompt_templates = [
            "Summarization the provided text.",
            "# Summary",
            "### Summary",
            "Summary of the text",
        ]
        # TODO: use these templates for training
        # analysis_templates = []
        # summary_templates = []

        def f(sample):
            return {
                "context": sample["chapter"],
                "prompt": "Summarize the provided text.",
                "response": sample["summary_text"],
            }

    elif ds_name == "gov_report":

        def f(sample):
            return {
                "context": sample["report"],
                "prompt": "Summarize the provided text.",
                "response": sample["summary"],
            }

    elif "wikitext" in ds_name:

        def f(sample):
            return {
                "context": sample["page"],
                "prompt": "PLAECHOLDER",
                "response": "PLAECHOLDER",
            }

    elif ds_name == "openmathintx-2":

        def f(sample):
            return {
                "context": sample["problem"],
                "prompt": sample["problem"],
                "response": sample["generated_solution"],
            }

    elif "gsm8k" in ds_name:

        def f(sample):
            return {
                "context": sample["question"],
                "prompt": sample["question"],
                "response": sample["answer"],
            }

    elif "opencoder-edu" in ds_name:

        def f(sample):
            return {
                "context": sample["instruction"],
                "prompt": sample["instruction"],
                "response": "```python\n" + sample["code"].strip() + "\n```",
            }

    if is_eval and (ds_name in EVAL_INTX_TEMPLATES) and not is_pretrain:
        prompt_template = EVAL_INTX_TEMPLATES[ds_name]

        def eval_intx_decorator(f):
            def g(sample):
                sample = f(sample)
                prompt_field = "prompt" if "prompt" in sample else "prompts"
                sample[prompt_field] = prompt_template.format(
                    input=sample[prompt_field]
                )
                return sample

            return g

        f = eval_intx_decorator(f)

    if not is_pretrain:

        def maybe_convert_to_list(f):
            def g(sample):
                sample = f(sample)
                if "prompt" in sample:
                    sample["prompts"] = [sample.pop("prompt")]
                if "response" in sample:
                    sample["responses"] = [sample.pop("response")]
                return sample

            return g

        f = maybe_convert_to_list(f)

    return f


def get_ds_kwargs(ds_name: str, split: str) -> dict[str, Any]:
    if ds_name.startswith("self_gen/"):
        # e.g., "self_gen/google/gemma-2-2b-it/pwc"
        base_model_name = "/".join(ds_name.split("/")[1:-1])
        base_ds = ds_name.split("/")[-1]
        files = glob(
            f"{SELF_GEN_DATA_DIR}/{base_model_name}/{base_ds}/{split}/*.parquet"
        )
        if not files:
            raise FileNotFoundError(
                f"No self-gen files found for base model {base_model_name} "
                f"in {SELF_GEN_DATA_DIR}/{base_model_name}/{base_ds}/"
            )
        kwargs = dict(path="parquet", data_files=files, split="train")
    elif (ds_name not in DS_KWARGS) or (split not in DS_KWARGS[ds_name]):
        kwargs = dict(path=ds_name, split=split)
        logger.warning(
            f"No dataset kwargs found for '{ds_name}' with split '{split}'.\n"
            f"Using default kwargs: {kwargs}"
        )
    else:
        kwargs = DS_KWARGS[ds_name][split]

    # # custom logic for slicing iterable datasets
    # take, skip = None, None
    # kw_split = kwargs["split"]
    # if ("[" in kw_split) and kw_split.endswith("]"):
    #     kwargs["split"], slice = kw_split.split("[")
    #     slice = slice.strip("]")
    #     skip = slice.split(":")[0]
    #     if skip:
    #         kwargs["skip"] = int(skip)
    #     take = slice.split(":")[1]
    #     if take:
    #         kwargs["take"] = int(take)
    return kwargs


def validate_columns(tokenized_ds):
    cols = ["input_ids", "attention_mask", "labels"]
    if "ctx_ids" in tokenized_ds.column_names:
        cols += ["ctx_ids", "ctx_attn_mask"]
    if "chat_ids" in tokenized_ds.column_names:
        cols += ["chat_ids", "chat_attn_mask", "chat_labels"]
    ref_cols = set(cols)
    assert set(tokenized_ds.column_names) == ref_cols, (
        f"Columns mismatch: {set(tokenized_ds.column_names)} != {ref_cols}"
    )


def len_filter(sample, max_length: int, keys: list[str]):
    m = [len(sample[k]) <= max_length for k in keys]
    return sum(m) == len(keys)


def len_filter_data(sample, max_length: int, keys: list[str]):
    data = sample["data"]
    indices = dict()
    for k in keys:
        indices[k] = {i for i, arr in enumerate(data[k]) if len(arr) <= max_length}

    all_indices = set.intersection(*indices.values())
    for k in data:
        data[k] = [data[k][i] for i in all_indices]

    return dict(data=data)


# def len_filter(ds, max_length: int, keys: list[str]):
#     for k in keys:
#         ds = ds.select([i for i, x in enumerate(ds[k]) if len(x) <= max_length])
#     return ds


# def filter_long_samples(samples):
#     return [len(ctx) < 10000 for ctx in samples["context"]]


# def filter_long_chat(samples):
#     return [len(chat) < 2**13 for chat in samples["chat"]]


def add_repeat_prompt_fn(samples: dict[str, list[str]], prob: float):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    for ctx in samples["context"]:
        if random.random() > prob:
            continue
        # Only process if the context is not already in the set
        if ctx in unique_contexts:
            continue
        # # remove too long contexts
        # if len(ctx) > 1000:
        #     continue
        unique_contexts.add(ctx)
        ctxs.append(ctx)
        responses.append([ctx])
        prompts.append([get_repeat_prompt()])

    logger.debug("Adding repeat prompt...")
    logger.debug(f"# unique contexts: {len(unique_contexts)}")

    return dict(
        context=ctxs + samples["context"],
        responses=responses + samples["responses"],
        prompts=prompts + samples["prompts"],
    )


# def add_negative_prompt_fn(samples):
#     unique_contexts = set()
#     ctxs, prompts, responses = [], [], []
#     keywords = [
#         "repeat",
#         "rephrase",
#         "summarize",
#         "rewrite",
#         "title",
#         "keyword",
#         "continuation",
#     ]

#     for ctx, prompt, response in zip(
#         samples["context"], samples["prompt"], samples["response"]
#     ):
#         if ctx in unique_contexts:
#             continue
#         unique_contexts.add(ctx)
#         if len(ctx) > 3000:
#             continue
#         if any(keyword in prompt for keyword in keywords):
#             # Skip samples where the prompt contains any of the specified keywords
#             continue

#         ctxs.append(ctx)
#         prompts.append(prompt)
#         responses.append(response)

#     logger.debug("Adding negative prompt...")
#     logger.debug(f"# unique contexts: {len(unique_contexts)}")

#     # remove one last sample if the number of samples is odd
#     if len(ctxs) % 2 != 0:
#         ctxs.pop()
#         prompts.pop()
#         responses.pop()

#     # to make sure that the negative prompt/response is not the same as the original
#     indices = list(np.random.permutation(len(ctxs))) + list(
#         np.random.permutation(len(ctxs))
#     )
#     neg_ctxs, neg_prompts, neg_responses = [], [], []
#     for idx in range(0, len(indices), 2):
#         i = indices[idx]
#         j = indices[idx + 1]
#         neg_ctxs.append(ctxs[i])
#         neg_prompts.append(ctxs[j] + "\n\n" + prompts[j])
#         neg_responses.append(responses[j])

#     return dict(
#         context=neg_ctxs + samples["context"],
#         prompt=neg_prompts + samples["prompt"],
#         response=neg_responses + samples["response"],
#     )


def filter_none(sample):
    # out = [True] * len(samples["context"])
    # for k in samples:
    #     for i, sample in enumerate(samples[k]):
    #         if not bool(sample):
    #             out[i] = False
    for v in sample.values():
        if v is None:
            return False
    return True


def load_and_process_dataset(
    ds_name: str,
    split: str,
    add_negative_prompt: bool,
    add_repeat_prompt: bool,
    repeat_prob: float,
    is_pretrain: bool,
    streaming: bool,
    num_proc: int,
):
    logger.info(f"Loading dataset {ds_name} with split {split}...")
    try:
        ds_kwargs = get_ds_kwargs(ds_name, split)
        skip = ds_kwargs.pop("skip", None)
        take = ds_kwargs.pop("take", None)
        # only load as stream if the dataset is local (parquet)
        load_as_stream = (
            ("fw_qa" in ds_name)
            and ("tiny" not in ds_name)
            and ("mini" not in ds_name)
            and streaming
            and ds_kwargs["path"] == "parquet"
        )
        ds = load_dataset(
            **ds_kwargs,
            trust_remote_code=True,
            streaming=load_as_stream,
        )
        if split == "train" and streaming and (not load_as_stream):
            # if the dataset is hosted on HF, we load it first then convert to iterable
            ds = ds.to_iterable_dataset(num_shards=128)
        if skip is not None:
            ds = ds.skip(skip)
        if take is not None:
            ds = ds.take(take)
    except ValueError as e:
        logger.warning(
            f"Failed to load dataset {ds_name} with split {split}. Error: {e}\nSkipping..."
        )
        return None
    cols_to_remove = [
        col
        for col in ds.column_names
        if col not in ["context", "prompts", "responses", "qas", "variation"]
    ]
    is_eval = split != "train"
    ds = ds.map(
        get_preprocessing_fn(ds_name, is_eval, is_pretrain),
        remove_columns=cols_to_remove,
        num_proc=num_proc,
    )
    # ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(
        filter_none,
        batched=False,
        num_proc=num_proc,
    )

    if split == "train":
        # TODO: drop samples or split into multiple contexts
        # for multi-lora training
        # ds = ds.filter(filter_long_samples, batched=True, num_proc=num_proc)
        if add_negative_prompt:
            # TODO: add explicit dataset for this
            # w/ qs that are answerable 0-shot
            # here we add the context to the prompt
            ds = ds.map(
                add_negative_prompt_fn,
                batched=True,
                batch_size=100_000,
                # num_proc=num_proc,
            )
        if (
            add_repeat_prompt
            and "context_numbers" not in ds_name
            and "pretrain" not in ds_name
        ):
            ds = ds.map(
                add_repeat_prompt_fn,
                fn_kwargs={"prob": repeat_prob},
                batched=True,
                batch_size=100_000,
                # num_proc=num_proc,
            )
    return ds


def get_tokenized_dataset(
    ds_name: str,
    split: str,
    base_model_max_len: int,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_kwargs: dict[str, Any],
    ctx_model_max_len: int,
    ctx_tokenizer: PreTrainedTokenizerBase,
    ctx_tokenizer_kwargs: dict[str, Any],
    add_ctx_to_chat: bool,
    add_repeat_prompt: bool,
    add_negative_prompt: bool,
    use_kl_loss: bool,
    repeat_prob: float = 0.0,  # only used when add_repeat_prompt is True
    set_format: str | None = None,
    streaming: bool = False,
) -> dict[str, Any]:
    assert not use_kl_loss, "KL loss is deprecated"
    if add_repeat_prompt:
        assert repeat_prob > 0, f"add_repeat_prompt is set but repeat_prob = 0"
    logger.info(f"Loading dataset {ds_name} with split {split}...")
    need_ctx_ids = not add_ctx_to_chat and bool(ctx_model_max_len)
    is_pretrain = "pretrain" in ds_name
    is_paraphrased = "aug_pretrain" in ds_name

    load_and_process_kwargs = dict(
        ds_name=ds_name,
        split=split,
        add_negative_prompt=add_negative_prompt,
        add_repeat_prompt=add_repeat_prompt,
        repeat_prob=repeat_prob,
        is_pretrain=is_pretrain,
        streaming=streaming,
    )
    tokenize_kwargs = dict(
        base_model_max_len=base_model_max_len,
        tokenizer_kwargs=tokenizer_kwargs,
        ctx_model_max_len=ctx_model_max_len,
        ctx_tokenizer_kwargs=ctx_tokenizer_kwargs,
        add_ctx_to_chat=add_ctx_to_chat,
        use_kl_loss=use_kl_loss,
        need_ctx_ids=need_ctx_ids,
        is_pretrain=is_pretrain,
        is_paraphrased=is_paraphrased,
        split=split,
        set_format=set_format,
    )

    all_kwargs = {**load_and_process_kwargs, **tokenize_kwargs}
    kwargs_str = json.dumps(all_kwargs)
    kwargs_str += repr(tokenizer) + repr(ctx_tokenizer)
    logger.debug(f"Tokenizing dataset with kwargs: {kwargs_str}")
    ds_hash = hashlib.sha256(kwargs_str.encode()).hexdigest()
    ds_path = f"{TRANSFORMED_DATA_DIR}/{ds_hash}"

    if path.exists(ds_path) and split == "train" and is_caching_enabled():
        # load the cached ds
        logger.info(f"Loaded tokenized dataset from {ds_path}")
        tokenized_ds = datasets.load_from_disk(ds_path)
        return tokenized_ds

    num_proc = None if streaming and split == "train" else 8
    ds = load_and_process_dataset(
        **load_and_process_kwargs,
        num_proc=num_proc,
    )
    logger.info(f"Constructing and tokenizing {ds_name} with {split} split...")
    tokenized_ds = construct_and_tokenize_ctx_qa(
        ds=ds,
        tokenizer=tokenizer,
        ctx_tokenizer=ctx_tokenizer,
        num_proc=num_proc,
        **tokenize_kwargs,
    )
    if split == "train" and is_caching_enabled():
        tokenized_ds.save_to_disk(ds_path, num_proc=num_proc)
        # force reload from disk for fingerprint consistency
        tokenized_ds = datasets.load_from_disk(ds_path)
    return tokenized_ds


def construct_and_tokenize_ctx_qa(
    base_model_max_len,
    tokenizer,
    tokenizer_kwargs,
    ctx_model_max_len,
    ctx_tokenizer,
    ctx_tokenizer_kwargs,
    add_ctx_to_chat,
    use_kl_loss,
    need_ctx_ids,
    is_pretrain,
    is_paraphrased,
    ds,
    split,
    set_format=None,
    num_proc=None,
):
    if is_pretrain:
        if is_paraphrased:
            ds = ds.map(
                build_paraphrase_pretrain,
                # batched=True,
                # batch_size=100_000,
                remove_columns=["variation"],
                num_proc=num_proc,
            )
        else:
            ds = ds.map(
                build_intx_pretrain,
                # batched=True,
                # batch_size=100_000,
                num_proc=num_proc,
            )
        tokenized_ds = ds.map(
            tokenize_pretrain,
            fn_kwargs={
                "tokenizer": tokenizer,
                "tokenizer_kwargs": tokenizer_kwargs,
                "add_eos_token": is_paraphrased,
            },
            batched=True,
            batch_size=100_000,
            # # num_proc=num_proc,
        )

    else:
        # convert everything into a new intermediate single column format
        # (required for tracking n_queries and
        # ejecting entire (ctx,questions,answers) triplet after filtering)
        # then convert back to columns after filtering

        # for sft + chat_model, we need to convert the dataset to chat format
        # add "data" field
        ds = ds.map(
            convert_ctx_prompt_response_to_messages,
            fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
            num_proc=num_proc,
            remove_columns=[col for col in ds.column_names if col != "context"],
        )
        # add input_ids, attention_mask, labels to "data"
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenized_ds = ds.map(
            get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer),
            batched=True,
            batch_size=100_000,
            # # num_proc=num_proc,
        )

    if split == "train":
        # base model cant handle these samples naturally
        # should skip
        # tokenized_ds = tokenized_ds.filter(
        #     len_filter,
        #     fn_kwargs={
        #         "max_length": base_model_max_len,
        #         "keys": ["input_ids", "labels"],
        #     },
        #     # batched=False,
        #     # batch_size=0,
        #     num_proc=num_proc,
        # )
        tokenized_ds = tokenized_ds.map(
            len_filter_data,
            fn_kwargs={
                "max_length": base_model_max_len,
                "keys": ["input_ids"],
            },
            num_proc=num_proc,
        )

    # for use_kl_loss, we need "chat_ids" and "chat_attn_mask"
    if use_kl_loss:
        raise NotImplementedError("KL loss deprecated")
        tokenized_ds = tokenized_ds.map(
            convert_ctx_prompt_response_to_messages,
            fn_kwargs={"add_ctx_to_chat": True},
            remove_columns=["messages"],
        )
        tokenized_ds = tokenized_ds.map(
            get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer),
            remove_columns=["chat"],
        )
        tokenized_ds = tokenized_ds.map(
            tokenize_chat_messages,
            fn_kwargs={
                "tokenizer": tokenizer,
                "mask_assistant_inputs": True,
                "tokenizer_kwargs": tokenizer_kwargs,
                "for_kl_loss": True,
            },
        )

    if need_ctx_ids:
        # tokenize the ctx_text to get ctx_ids and ctx_attn_mask
        os.environ["TOKENIZERS_PARALLELISM"] = "true"
        tokenized_ds = tokenized_ds.map(
            tokenize_ctx_text,
            fn_kwargs={"tokenizer": ctx_tokenizer},
            batched=True,
            batch_size=100_000,
            # # num_proc=num_proc,
        )
        if split == "train":
            # TODO: do something if ctx length is longer than the ctx model length
            # e.g., drop or split for multi-lora training
            tokenized_ds = tokenized_ds.filter(
                len_filter,
                fn_kwargs={"max_length": ctx_model_max_len, "keys": ["ctx_ids"]},
                num_proc=num_proc,
            )

    if not is_pretrain:
        if split == "train":
            tokenized_ds = tokenized_ds.map(
                unpack_data,
                num_proc=num_proc,
                remove_columns=["data", "context"],
            )
        else:
            cols_to_remove = ["data", "context"]
            if "ctx_ids" in tokenized_ds.column_names:
                cols_to_remove += ["ctx_ids", "ctx_attn_mask"]
            tokenized_ds = tokenized_ds.map(
                unpack_data_eval,
                num_proc=num_proc,
                remove_columns=cols_to_remove,
                batched=True,
                batch_size=100_000,
            )
            tokenized_ds = tokenized_ds.map(
                truncate_middle_if_too_long,
                fn_kwargs={
                    "max_length": base_model_max_len,
                    "columns": ["input_ids", "attention_mask", "labels"],
                    "max_new_tokens": 256,
                },
            )
            tokenized_ds = tokenized_ds.map(
                add_length_info,
                fn_kwargs={"columns": ["input_ids"]},
            )
            if "ctx_ids" in tokenized_ds.column_names:
                tokenized_ds = tokenized_ds.map(
                    truncate_middle_if_too_long,
                    fn_kwargs={
                        "max_length": ctx_model_max_len,
                        "columns": ["ctx_ids", "ctx_attn_mask"],
                        # cxt encoder doesnt need to add new_tokens
                        "max_new_tokens": 0,
                    },
                )
                tokenized_ds = tokenized_ds.map(
                    add_length_info,
                    fn_kwargs={"columns": ["ctx_ids"]},
                )

    if is_pretrain:
        tokenized_ds = tokenized_ds.remove_columns(["context", "text"])
        if not is_paraphrased:
            tokenized_ds = tokenized_ds.remove_columns(["qas"])

    if set_format:
        tokenized_ds.set_format(type=set_format)

    # validate_columns(tokenized_ds)

    return tokenized_ds


def build_paraphrase_pretrain(samples: dict[str, list[str]]):
    # use "context" to predict "variation" as vice versa
    return {
        "text": samples["context"] + samples["variation"],
        "context": samples["variation"] + samples["context"],
    }


def build_intx_pretrain(sample: dict[str, str]):
    return {"text": sample["context"] + sample["qas"]}


def get_sft_prompt_formatting_fn(
    sft_mode: TRAINING_TASK,
    tokenizer: PreTrainedTokenizerBase,
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get a function that formats examples for supervised fine-tuning.

    Args:
        sft_mode: The training task type
        tokenizer: The tokenizer to use for chat template application

    Returns:
        A function that takes a training example and returns formatted data

    Raises:
        NotImplementedError: If sft_mode is not COMPLETION or tokenizer has no chat template
    """
    if sft_mode != TRAINING_TASK.COMPLETION:
        raise NotImplementedError(
            f"Training task {sft_mode} not supported. "
            "Only completion is supported for now."
        )

    if tokenizer.chat_template is None:
        raise NotImplementedError(
            "Only chat models + SFT are supported. "
            "Training with pre-training data is not supported yet."
        )

    def f_intx(samples):
        # flatten all the messages into a list
        # tokenize, the pack back correctly
        messages_list = [x for x in samples["data"]]

        n_queries = [len(x) for x in messages_list]
        messages = concat_list(messages_list)
        logger.info(f"Tokenizing {len(messages)} messages...")
        tokens = tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_special_token=False,
            truncation=False,
            add_generation_prompt=False,
            return_assistant_tokens_mask=True,
            return_dict=True,
        )
        labels = []
        for tok_ids, masks in zip(tokens["input_ids"], tokens["assistant_masks"]):
            o = [id_ if mask else IGNORE_INDEX for id_, mask in zip(tok_ids, masks)]
            labels.append(o)
        del tokens["assistant_masks"]
        tokens["labels"] = labels
        per_ctx_tokens = []
        i = 0
        for n in n_queries:
            per_ctx_tokens.append(
                {
                    "input_ids": tokens["input_ids"][i : i + n],
                    "attention_mask": tokens["attention_mask"][i : i + n],
                    "labels": tokens["labels"][i : i + n],
                }
            )
            i += n

        return dict(data=per_ctx_tokens)

    # return f if not apply_chat_template_fn is not None else f_intx
    return f_intx


def convert_ctx_prompt_response_to_messages(
    example: dict[str, Any],
    add_ctx_to_chat: bool,
) -> dict[str, Any]:
    """
    Convert context/prompt/response format to chat messages format.

    Args:
        example: Dictionary containing 'prompt' and 'response' keys
        add_ctx_to_chat: Whether to prepend context to the user message

    Returns:
        Dictionary with added 'messages' key containing chat format

    Raises:
        ValueError: If 'prompt' or 'response' keys are missing
    """
    prompt_field = "prompts"
    res_field = "responses"

    if prompt_field not in example or res_field not in example:
        raise ValueError(
            f"'{prompt_field}' and '{res_field}' are required. Got: {example}"
        )

    system_msg = ""
    if "system_message" in example:
        system_msg = example["system_message"].strip()

    data = []
    for prompt, response in zip(example[prompt_field], example[res_field]):
        user_msg = prompt.strip()
        if add_ctx_to_chat:
            user_msg = example["context"].strip() + "\n\n" + user_msg

        data.append(
            [
                {"role": "system", "content": system_msg.strip()},
                {"role": "user", "content": user_msg.strip()},
                {"role": "assistant", "content": response.strip()},
            ]
        )

    return dict(data=data)


def unpack_data(sample):
    return {k: sample["data"][k] for k in sample["data"]}


def unpack_data_eval(samples):
    # n_queries always == 1 for eval
    data = samples["data"]
    out = dict(input_ids=[], attention_mask=[], labels=[])
    if "ctx_ids" in samples:
        out["ctx_ids"] = []
        out["ctx_attn_mask"] = []
    for i, d in enumerate(data):
        for tokens in zip(
            d["input_ids"],
            d["attention_mask"],
            d["labels"],
        ):
            if "ctx_ids" in samples:
                out["ctx_ids"].append(samples["ctx_ids"][i])
                out["ctx_attn_mask"].append(samples["ctx_attn_mask"][i])
            out["input_ids"].append(tokens[0])
            out["attention_mask"].append(tokens[1])
            out["labels"].append(tokens[2])
    return out


# adapted from https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
def get_assistant_start_end_indices(
    messages: list[dict[str, str]],
    tokenizer: PreTrainedTokenizerBase,
    conversation_ids: list[int],
) -> list[tuple[int, int]]:
    """
    Get the start and end indices of assistant messages in conversation text using tokenized messages.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        tokenizer: Tokenizer used for tokenization
        conversation_ids: Tokenized conversation

    Returns:
        List of (start, end) index tuples for assistant messages in the tokenized conversation
    """
    assistant_ranges = []
    current_idx = 0

    for idx, message in enumerate(messages):
        # if message["role"] == "assistant":
        # Tokenize the assistant message content
        tokenized_message = tokenizer.encode(
            message["content"], add_special_tokens=False
        )
        # Find the start index of the assistant message in the tokenized conversation
        for i in range(current_idx, len(conversation_ids)):
            if conversation_ids[i : i + len(tokenized_message)] == tokenized_message:
                start_idx = i
                end_idx = i + len(tokenized_message)
                current_idx = end_idx
                if message["role"] == "assistant":
                    if idx == (len(messages) - 1):
                        # include everything after the last assistant message
                        assistant_ranges.append((start_idx, len(conversation_ids)))
                    else:
                        assistant_ranges.append((start_idx, end_idx))
                break

    return assistant_ranges


# def get_masked_labels(
#     conversation_ids: dict[str, list[Any]],
#     assistant_ranges: list[tuple[int, int]],
# ) -> Iterator[int]:
#     """
#     Generate masked labels for conversation, masking non-assistant tokens.
#     NOTE: This will also includes extra tokens between assistant and user messages
#     in multi-turn conversations.
#     E.g., {assistant_msg} <|eot_id|><|start_header_id|>user<|end_header_id|> {user_msg}
#     for Llama 3 models.

#     Args:
#         conversation_ids: Dictionary with tokenized conversation info
#         assistant_ranges: List of (start, end) indices for assistant messages

#     Yields:
#         Token ID or IGNORE_INDEX for each position
#     """
#     for id_, (id_s, id_e) in list(
#         zip(conversation_ids["input_ids"], conversation_ids["offset_mapping"])
#     ):
#         if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
#             yield id_
#         else:
#             yield IGNORE_INDEX


# def tokenize_chat_messages(
#     example: dict[str, Any],
#     tokenizer: PreTrainedTokenizerBase,
#     mask_assistant_inputs: bool = True,
#     for_kl_loss: bool = False,
#     tokenizer_kwargs: dict[str, Any] | None = None,
# ) -> dict[str, list[int]]:
#     """
#     Tokenize chat messages and optionally mask non-assistant tokens.

#     Args:
#         example: Dictionary containing 'chat' and 'messages' keys
#         tokenizer: Tokenizer to use
#         mask_assistant_inputs: Whether to mask non-assistant tokens
#         for_kl_loss: change the column names to "chat_ids" and "chat_attn_mask"
#         tokenizer_kwargs: Additional arguments to pass to tokenizer

#     Returns:
#         Dictionary containing tokenized inputs and labels
#     """
#     # should be used only with chat models
#     text = example["chat"]
#     messages = example["messages"]
#     n_response = len([m for m in messages if m["role"] == "assistant"])
#     if n_response != 1:
#         raise ValueError(f"Expected 1 assistant response. Got {n_response}.")
#     conversation_ids = tokenizer(
#         text,
#         # return_offsets_mapping=mask_assistant_inputs,
#         add_special_tokens=False,
#         truncation=False,
#         **(tokenizer_kwargs or {}),
#     )
#     # if tokenizer_kwargs and (
#     #     len(conversation_ids["input_ids"]) >= tokenizer_kwargs["max_length"]
#     # ):
#     #     raise ValueError(
#     #         f"Conversation length {len(conversation_ids['input_ids'])} exceeds max length {tokenizer_kwargs['max_length']}"
#     #     )

#     if mask_assistant_inputs:
#         assistant_ranges = get_assistant_start_end_indices(
#             messages, tokenizer, conversation_ids["input_ids"]
#         )
#         # labels = get_masked_labels(conversation_ids, assistant_ranges)
#         labels = [
#             id_ if any(s <= i < e for s, e in assistant_ranges) else IGNORE_INDEX
#             for i, id_ in enumerate(conversation_ids["input_ids"])
#         ]
#         conversation_ids["labels"] = labels
#         # del conversation_ids["offset_mapping"]
#     else:
#         conversation_ids["labels"] = conversation_ids["input_ids"]
#     if for_kl_loss:
#         conversation_ids["chat_ids"] = conversation_ids.pop("input_ids")
#         conversation_ids["chat_attn_mask"] = conversation_ids.pop("attention_mask")
#         conversation_ids["chat_labels"] = conversation_ids.pop("labels")
#     return conversation_ids


def tokenize_pretrain(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_kwargs: dict[str, Any] | None = None,
    add_eos_token: bool = False,
):
    """
    Tokenize text for pre-training tasks.

    Args:
        samples: Dictionary containing 'text' key
        tokenizer: Tokenizer to use
        tokenizer_kwargs: Additional arguments to pass to tokenizer

    Returns:
        Dictionary containing tokenized inputs and labels
    """
    add_bos_val = tokenizer.add_bos_token
    add_eos_val = tokenizer.add_eos_token
    tokenizer.add_bos_token = True
    tokenizer.add_eos_token = add_eos_token
    tokens = tokenizer(
        samples["text"],
        add_special_tokens=True,
        truncation=False,
        **(tokenizer_kwargs or {}),
    )
    # if tokenizer.bos_token_id is not None:
    #     for i in range(len(tokens["input_ids"])):
    #         # add bos
    #         tokens["input_ids"][i] = [tokenizer.bos_token_id] + tokens["input_ids"][i]
    #         tokens["attention_mask"][i] = [1] + tokens["attention_mask"][i]
    tokenizer.add_bos_token = add_bos_val
    tokenizer.add_eos_token = add_eos_val

    tokens["labels"] = tokens["input_ids"]
    return tokens


def add_length_info(sample: dict[str, any], columns: list[str]) -> dict[str, any]:
    return {f"{k}_len": len(sample[k]) for k in columns}
    # for col in columns:
    #     samples[f"{col}_len"] = [len(t) for t in samples[col]]
    # return samples


def truncate_middle_if_too_long(
    sample: dict[str, any],
    max_length: int,
    columns: list[str],
    max_new_tokens: int = 256,
) -> dict[str, any]:
    """
    Truncate the middle of a list of tokens to fit within a maximum length.

    Args:
        tokens: List of token IDs
        max_length: Maximum length for the truncated tokens

    Returns:
        List of truncated token IDs
    """
    max_new_tokens_half = max_new_tokens // 2
    # leave max_new_tokens for generation
    half = max_length // 2 - max_new_tokens_half
    for col in columns:
        # if len(samples[col]) <= max_length:
        #     return samples[col]
        t = sample[col]
        sample[col] = t[:half] + t[-half:] if len(t) > max_length else t
    return sample


def tokenize_ctx_text(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    if tokenizer.chat_template:
        tokenized_text = tokenizer.apply_chat_template(
            [
                [
                    {"role": "system", "content": ""},
                    {"role": "user", "content": text.strip()},
                ]
                for text in samples["context"]
            ],
            tokenize=True,
            add_generation_prompt=True,
            padding=False,
            truncation=False,
            add_special_tokens=False,
            return_dict=True,
        )
        # special tokens are already added by the chat template
        # tokenized_text = tokenizer(txt, truncation=False, add_special_tokens=False)
    else:
        raise NotImplementedError("Only support chat models.")
        # txt = samples["context"]
        # tokenized_text = tokenizer(txt, truncation=False, add_special_tokens=False)
    ctx_ids = tokenized_text["input_ids"]
    ctx_attn_mask = tokenized_text["attention_mask"]
    return dict(ctx_ids=ctx_ids, ctx_attn_mask=ctx_attn_mask)


def pack(
    ds: Dataset,
    max_packed_inp_len: int,
    max_packed_ctx_len: int,
    max_packed_size: int,
    num_proc: int = 0,
):
    # TODO: handle the new data format
    # this would generate a giant cache file for the already concat'd + packed ds
    kwargs = dict(
        max_packed_inp_len=max_packed_inp_len,
        max_packed_ctx_len=max_packed_ctx_len,
        max_packed_size=max_packed_size,
    )
    ds_hash = hashlib.sha256(
        (ds._fingerprint + json.dumps(kwargs)).encode()
    ).hexdigest()
    ds_path = f"{TRANSFORMED_DATA_DIR}/{ds_hash}"
    logger.info(
        f"Packing dataset {ds_hash} with max_packed_inp_len={max_packed_inp_len} and max_packed_ctx_len={max_packed_ctx_len}"
    )
    if path.exists(ds_path) and is_caching_enabled():
        logger.info(f"Loading a cached packed dataset for {ds_path}")
        return datasets.load_from_disk(ds_path)
    else:
        ds = ds.map(
            pack_batch,
            fn_kwargs={
                "max_packed_inp_len": max_packed_inp_len,
                "max_packed_ctx_len": max_packed_ctx_len,
                "max_packed_size": max_packed_size,
            },
            batched=True,
            batch_size=250_000,
            num_proc=num_proc,
            remove_columns=ds.column_names,
        )

        ds.save_to_disk(ds_path, num_proc=num_proc)
    return ds


if __name__ == "__main__":
    from transformers import AutoTokenizer

    # model_name = "meta-llama/Llama-3.2-1B-Instruct"
    # model_name = "meta-llama/Llama-2-7b-chat-hf"
    model_name = "google/gemma-2-2b-it"
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hello!"},
        # {"role": "user", "content": "Not too bad"},
        # {"role": "assistant", "content": "Cooooooool"},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    print(chat)
    model_inputs = tokenize_chat_messages(
        {"chat": chat, "messages": messages},
        tokenizer,
        # for_kl_loss=True,
    )

    print(tokenizer(chat, add_special_tokens=False))
    print(model_inputs)
