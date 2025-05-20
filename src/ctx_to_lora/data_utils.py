import logging
import hashlib
import json
import random
from functools import partial
from os import path
from glob import glob
from typing import Any, Callable, Iterator, Optional


import datasets
import numpy as np
from datasets import load_dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from ctx_to_lora.training_utils import TRAINING_TASK
from ctx_to_lora.data_definitions import (
    DS_KWARGS,
    EVAL_INTX_TEMPLATES,
    TRANSFORMED_DATA_DIR,
    IGNORE_INDEX,
    CLOSED_QA_INTX_TEMPLATES,
)

logger = logging.getLogger()

# TODO: add continued pre-training pipeline


def closed_qa_prompting(prompt: str):
    template = random.choice(CLOSED_QA_INTX_TEMPLATES)
    return template.format(input=prompt)


# TODO: dont use closed_qa_prompting when training on self-generated responses
def get_preprocessing_fn(
    ds_name: str, is_eval: bool
) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get preprocessing function for a specific dataset.

    Args:
        ds_name: Name of the dataset

    Returns:
        A preprocessing function that takes and returns a dictionary
    """
    f = lambda x: x

    if ds_name.startswith("pwc"):

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

    elif "squad" in ds_name:

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
                "response": ", ".join(sample["answers_spans"]["spans"]),
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

    elif ds_name.startswith("longbench"):

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["input"],
                "response": sample["answers"][0],
            }

    elif "natural_questions" in ds_name:
        pass
        # ctx = " ".join(
        #     [
        #         token
        #         for is_html, token in zip(
        #             ds[0]["document"]["tokens"]["is_html"],
        #             ds[0]["document"]["tokens"]["token"],
        #         )
        #         if not is_html
        #     ]
        # )
    elif ds_name == "triviaqa_retrieved":
        # maybe needed for training (negative sample for training)
        # def f(sample):
        #     ctx = sample["entity_page"]["wiki_context"]
        #     if not ctx:
        #         return None

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["prompt"],
                "response": sample["answer"],
            }

    if is_eval and ds_name in EVAL_INTX_TEMPLATES:
        prompt_template = EVAL_INTX_TEMPLATES[ds_name]

        def eval_intx_decorator(f):
            def g(sample):
                sample = f(sample)
                sample["prompt"] = prompt_template.format(input=sample["prompt"])
                return sample

            return g

    return eval_intx_decorator(f) if is_eval else f


def get_ds_kwargs(ds_name: str, split: str) -> dict[str, Any]:
    if (ds_name not in DS_KWARGS) or (split not in DS_KWARGS[ds_name]):
        kwargs = dict(path=ds_name, split=split)
        logger.warning(
            f"No dataset kwargs found for '{ds_name}' with split '{split}'.\n"
            f"Using default kwargs: {kwargs}"
        )
        # return kwargs
    else:
        # return DS_KWARGS[ds_name][split]
        kwargs = DS_KWARGS[ds_name][split]

    # custom logic for slicing iterable datasets
    take, skip = None, None
    kw_split = kwargs["split"]
    if ("[" in kw_split) and kw_split.endswith("]"):
        kwargs["split"], slice = kw_split.split("[")
        slice = slice.strip("]")
        skip = slice.split(":")[0]
        if skip:
            kwargs["skip"] = int(skip)
        take = slice.split(":")[1]
        if take:
            kwargs["take"] = int(take)
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


def filter_long_samples(samples):
    return [len(ctx) < 10000 for ctx in samples["context"]]


def filter_long_chat(samples):
    return [len(chat) < 2**13 for chat in samples["chat"]]


def add_repeat_prompt_fn(samples):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    for ctx, prompt, response in zip(
        samples["context"], samples["prompt"], samples["response"]
    ):
        # Only process if the context is not already in the set
        if ctx in unique_contexts:
            continue
        # remove too long contexts
        if len(ctx) > 1000:
            continue
        unique_contexts.add(ctx)
        ctxs.append(ctx)
        responses.append(ctx)
        prompts.append("Repeat the text above.")

    logger.debug("Adding repeat prompt...")
    logger.debug(f"# unique contexts: {len(unique_contexts)}")

    return dict(
        context=ctxs + samples["context"],
        response=responses + samples["response"],
        prompt=prompts + samples["prompt"],
    )


def add_negative_prompt_fn(samples):
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

    for ctx, prompt, response in zip(
        samples["context"], samples["prompt"], samples["response"]
    ):
        if ctx in unique_contexts:
            continue
        unique_contexts.add(ctx)
        if len(ctx) > 3000:
            continue
        if any(keyword in prompt for keyword in keywords):
            # Skip samples where the prompt contains any of the specified keywords
            continue

        ctxs.append(ctx)
        prompts.append(prompt)
        responses.append(response)

    logger.debug(f"Adding negative prompt...")
    logger.debug(f"# unique contexts: {len(unique_contexts)}")

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

    return dict(
        context=neg_ctxs + samples["context"],
        prompt=neg_prompts + samples["prompt"],
        response=neg_responses + samples["response"],
    )


def filter_none(samples):
    out = [True] * len(samples["context"])
    for k in samples:
        for i, sample in enumerate(samples[k]):
            if not bool(sample):
                out[i] = False
    return out


def _load_and_process_dataset(
    ds_name: str,
    split: str,
    add_negative_prompt: bool,
    add_repeat_prompt: bool,
    streaming: bool,
    ds_path: str,
    num_proc: int,
):
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
        col for col in ds.column_names if col not in ["context", "prompt", "response"]
    ]
    is_eval = split != "train"
    ds = ds.map(
        get_preprocessing_fn(ds_name, is_eval),
        remove_columns=cols_to_remove,
        num_proc=num_proc,
    )
    # ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(filter_none, batched=True, num_proc=num_proc)

    if split == "train":
        # TODO: drop samples or split into multiple contexts
        ds = ds.filter(filter_long_samples, batched=True, num_proc=num_proc)
        if add_negative_prompt:
            # TODO: make explicit dataset for this
            ds = ds.map(
                add_negative_prompt_fn,
                batched=True,
                batch_size=100_000,
                num_proc=num_proc,
            )
        if add_repeat_prompt and "context_numbers" not in ds_name:
            # TODO: convert to AE/LM
            ds = ds.map(
                add_repeat_prompt_fn,
                batched=True,
                batch_size=100_000,
                num_proc=num_proc,
            )
        ds.save_to_disk(ds_path, num_proc=num_proc)
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
    set_format: Optional[str] = None,
    streaming: bool = False,
) -> dict[str, Any]:
    # TODO: update hash function to include max_len
    # TODO: max_len could be included in the kwargs
    assert not use_kl_loss, "KL loss is deprecated"
    logger.debug(f"Loading dataset {ds_name} with split {split}...")
    need_ctx_ids = not add_ctx_to_chat and ctx_model_max_len

    load_and_process_kwargs = dict(
        ds_name=ds_name,
        split=split,
        add_negative_prompt=add_negative_prompt,
        add_repeat_prompt=add_repeat_prompt,
        streaming=streaming,
    )
    num_proc = None if streaming and split == "train" else 8

    ds_hash = hashlib.sha256(json.dumps(load_and_process_kwargs).encode()).hexdigest()
    ds_path = f"{TRANSFORMED_DATA_DIR}/{ds_hash}"

    if path.exists(ds_path) and split == "train":
        # load the cached ds
        logger.info(f"Loaded processed dataset from {ds_path}")
    else:
        logger.info(f"Loading dataset {ds_name} with split {split}...")
        ds = _load_and_process_dataset(
            **load_and_process_kwargs,
            ds_path=ds_path,
            num_proc=num_proc,
        )
    if split == "train":
        # force loading the cached version for newly created ds
        ds = datasets.load_from_disk(ds_path)

    tokenized_ds = construct_and_tokenize_ctx_qa(
        base_model_max_len,
        tokenizer,
        tokenizer_kwargs,
        ctx_model_max_len,
        ctx_tokenizer,
        ctx_tokenizer_kwargs,
        add_ctx_to_chat,
        use_kl_loss,
        need_ctx_ids,
        ds,
        split,
        set_format,
        num_proc,
    )
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
    ds,
    split,
    set_format=None,
    num_proc=None,
):
    is_eval = split != "train"
    # TODO: update hash function to include max_len
    # TODO: max_len could be included in the kwargs
    kwargs = dict(
        tokenizer=repr(tokenizer),
        tokenizer_kwargs=json.dumps(tokenizer_kwargs),
        ctx_tokenizer=repr(ctx_tokenizer),
        ctx_tokenizer_kwargs=json.dumps(ctx_tokenizer_kwargs),
        add_ctx_to_chat=add_ctx_to_chat,
        use_kl_loss=use_kl_loss,
        need_ctx_ids=need_ctx_ids,
        ds=ds._fingerprint,
        # TODO: add split (after moving to continued pre-training pipeline)
        set_format=set_format,
    )
    kwargs_str = json.dumps(kwargs)
    logger.debug(f"Tokenizing dataset with kwargs: {kwargs_str}")
    ds_hash = hashlib.sha256(kwargs_str.encode()).hexdigest()
    ds_path = f"{TRANSFORMED_DATA_DIR}/{ds_hash}"
    if path.exists(ds_path) and split == "train":
        # load the cached ds
        logger.info(f"Loaded tokenized dataset from {ds_path}")
        ds = datasets.load_from_disk(ds_path)
        return ds
    # for sft + chat_model, we need to convert the dataset to chat format
    # add "messages" field
    ds = ds.map(
        convert_ctx_prompt_response_to_messages,
        fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
        num_proc=num_proc,
    )
    # add "chat" field
    ds = ds.map(
        get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer),
        batched=True,
        batch_size=100_000,
        num_proc=num_proc,
    )
    # tokenize the chat + mask the assistant inputs
    # TODO: drop samples or split into multiple contexts
    if split == "train":
        ds = ds.filter(
            filter_long_chat,
            batched=True,
            batch_size=100_000,
            num_proc=num_proc,
        )

    # add "input_ids", "attention_mask", "labels"
    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={
            "tokenizer": tokenizer,
            "mask_assistant_inputs": True,
            "tokenizer_kwargs": tokenizer_kwargs,
        },
        num_proc=num_proc,
    )
    if split == "train":
        # drop?
        # currently the above function would raise if the chat is too long
        ...
    else:
        tokenized_ds = tokenized_ds.map(
            truncate_middle_if_too_long,
            fn_kwargs={
                "max_length": base_model_max_len,
                "columns": ["input_ids", "attention_mask"],
            },
            batched=True,
            batch_size=100_000,
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
        tokenized_ds = tokenized_ds.map(
            tokenize_ctx_text,
            fn_kwargs={"tokenizer": ctx_tokenizer},
            batched=True,
            batch_size=100_000,
            num_proc=num_proc,
        )
        if split == "train":
            # TODO: do something if ctx length is longer than the ctx model length
            # e.g., drop or split for multi-lora training
            ...

        else:
            # truncate in the middle
            tokenized_ds = tokenized_ds.map(
                truncate_middle_if_too_long,
                fn_kwargs={
                    "max_length": ctx_model_max_len,
                    "columns": ["ctx_ids", "ctx_attn_mask"],
                },
                batched=True,
                batch_size=100_000,
                num_proc=num_proc,
            )
    tokenized_ds = tokenized_ds.remove_columns(
        ["messages", "chat", "context", "prompt", "response"]
    )
    if set_format:
        tokenized_ds.set_format(type=set_format)
    # if isinstance(tokenized_ds, IterableDataset):
    #     # the columns are unknown when using streaming dataset
    #     tokenized_ds = tokenized_ds._resolve_features()
    # validate_columns(tokenized_ds)
    if split == "train":
        tokenized_ds.save_to_disk(ds_path, num_proc=num_proc)
        del tokenized_ds
        return datasets.load_from_disk(ds_path)
    return tokenized_ds


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
    # TODO: add support for causal_lm training (for pre-training data)
    # TODO: add support for recon training (for pre-training data)
    # TODO: add support for non-chat models

    # def f(example):
    #     output_texts = (
    #         dict(text=[]) if sft_mode == "causal_lm" else dict(prompt=[], response=[])
    #     )
    #     df = pd.DataFrame(dict(example))
    #     for i, inp_txt in df.iterrows():
    #         if sft_mode == "causal_lm":
    #             text = metadata["text_template"].format(**inp_txt)
    #             output_texts["text"].append(text)
    #         elif sft_mode == "completion":
    #             prompt = metadata["user_prompt_template"].format(**inp_txt)
    #             output_texts["prompt"].append(prompt)
    #             output_texts["response"].append(str(inp_txt[metadata["response_field"]]))
    #     return output_texts

    def f_intx(example):
        chat_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        return dict(chat=chat_text)

    # return f if not apply_chat_template_fn is not None else f_intx
    return f_intx


def convert_ctx_prompt_response_to_messages(
    example: dict[str, Any],
    add_ctx_to_chat: bool = True,
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
    if "prompt" not in example or "response" not in example:
        raise ValueError(f"'prompt' and 'response' are required. Got: {example}")

    system_msg = ""
    if "system_message" in example:
        system_msg = example["system_message"].strip()

    user_msg = example["prompt"].strip()
    if add_ctx_to_chat:
        user_msg = example["context"].strip() + "\n\n" + user_msg.strip()

    messages = [
        {"role": "system", "content": system_msg.strip()},
        {"role": "user", "content": user_msg.strip()},
        {"role": "assistant", "content": example["response"].strip()},
    ]

    return dict(messages=messages)


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


def get_masked_labels(
    conversation_ids: dict[str, list[Any]],
    assistant_ranges: list[tuple[int, int]],
) -> Iterator[int]:
    """
    Generate masked labels for conversation, masking non-assistant tokens.
    NOTE: This will also includes extra tokens between assistant and user messages
    in multi-turn conversations.
    E.g., {assistant_msg} <|eot_id|><|start_header_id|>user<|end_header_id|> {user_msg}
    for Llama 3 models.

    Args:
        conversation_ids: Dictionary with tokenized conversation info
        assistant_ranges: List of (start, end) indices for assistant messages

    Yields:
        Token ID or IGNORE_INDEX for each position
    """
    for id_, (id_s, id_e) in list(
        zip(conversation_ids["input_ids"], conversation_ids["offset_mapping"])
    ):
        if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
            yield id_
        else:
            yield IGNORE_INDEX


def tokenize_chat_messages(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
    mask_assistant_inputs: bool = True,
    for_kl_loss: bool = False,
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[int]]:
    """
    Tokenize chat messages and optionally mask non-assistant tokens.

    Args:
        example: Dictionary containing 'chat' and 'messages' keys
        tokenizer: Tokenizer to use
        mask_assistant_inputs: Whether to mask non-assistant tokens
        for_kl_loss: change the column names to "chat_ids" and "chat_attn_mask"
        tokenizer_kwargs: Additional arguments to pass to tokenizer

    Returns:
        Dictionary containing tokenized inputs and labels
    """
    # should be used only with chat models
    text = example["chat"]
    messages = example["messages"]
    n_response = len([m for m in messages if m["role"] == "assistant"])
    if n_response != 1:
        raise ValueError(f"Expected 1 assistant response. Got {n_response}.")
    conversation_ids = tokenizer(
        text,
        # return_offsets_mapping=mask_assistant_inputs,
        add_special_tokens=False,
        truncation=False,
        **(tokenizer_kwargs or {}),
    )
    if tokenizer_kwargs and (
        len(conversation_ids["input_ids"]) >= tokenizer_kwargs["max_length"]
    ):
        raise ValueError(
            f"Conversation length {len(conversation_ids['input_ids'])} exceeds max length {tokenizer_kwargs['max_length']}"
        )

    if mask_assistant_inputs:
        assistant_ranges = get_assistant_start_end_indices(
            messages, tokenizer, conversation_ids["input_ids"]
        )
        # labels = get_masked_labels(conversation_ids, assistant_ranges)
        labels = [
            id_ if any(s <= i < e for s, e in assistant_ranges) else IGNORE_INDEX
            for i, id_ in enumerate(conversation_ids["input_ids"])
        ]
        conversation_ids["labels"] = labels
        # del conversation_ids["offset_mapping"]
    else:
        conversation_ids["labels"] = conversation_ids["input_ids"]
    if for_kl_loss:
        conversation_ids["chat_ids"] = conversation_ids.pop("input_ids")
        conversation_ids["chat_attn_mask"] = conversation_ids.pop("attention_mask")
        conversation_ids["chat_labels"] = conversation_ids.pop("labels")
    return conversation_ids


def truncate_middle_if_too_long(
    samples: dict[str, any],
    max_length: int,
    columns: list[str],
) -> dict[str, any]:
    """
    Truncate the middle of a list of tokens to fit within a maximum length.

    Args:
        tokens: List of token IDs
        max_length: Maximum length for the truncated tokens

    Returns:
        List of truncated token IDs
    """
    half = max_length // 2
    for col in columns:
        # if len(samples[col]) <= max_length:
        #     return samples[col]
        samples[col] = [
            t[:half] + t[-half:] if len(t) > max_length else t for t in samples[col]
        ]
    return samples


def tokenize_ctx_text(
    samples: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    if tokenizer.chat_template:
        txt = tokenizer.apply_chat_template(
            [[{"role": "user", "content": text.strip()}] for text in samples["context"]],
            tokenize=False,
            add_generation_prompt=False,
        )
        # special tokens are already added by the chat template
        tokenized_text = tokenizer(txt, truncation=False, add_special_tokens=False)
    else:
        txt = samples["context"]
        tokenized_text = tokenizer(txt, truncation=False)

    ctx_ids = tokenized_text["input_ids"]
    ctx_attn_mask = tokenized_text["attention_mask"]
    return dict(ctx_ids=ctx_ids, ctx_attn_mask=ctx_attn_mask)


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
