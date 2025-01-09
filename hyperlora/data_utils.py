import logging
import random
import numpy as np
from copy import copy
from typing import Any, Callable, Dict, Iterator, List, Literal, Optional, Tuple, Union

import pandas as pd
from datasets import load_dataset
from training_utils import TRAINING_TASK
from transformers import PreTrainedTokenizerBase

IGNORE_INDEX = -100

logger = logging.getLogger()


def validate_columns(tokenized_ds):
    cols = ["input_ids", "attention_mask", "labels"]
    if "ctx_ids" in tokenized_ds.column_names:
        cols += ["ctx_ids", "ctx_attn_mask"]
    ref_cols = set(cols)
    assert (
        set(tokenized_ds.column_names) == ref_cols
    ), f"Columns mismatch: {set(tokenized_ds.column_names)} != {ref_cols}"


def filter_long_samples(samples):
    return [len(ctx) < 10000 for ctx in samples["context"]]


def add_repeat_prompt_fn(samples):
    unique_contexts = set()
    ctxs, prompts, responses = [], [], []
    for ctx, prompt, response in zip(
        samples["context"], samples["prompt"], samples["response"]
    ):
        # Only process if the context is not already in the set
        if ctx in unique_contexts:
            continue

        unique_contexts.add(ctx)
        ctxs.append(ctx)
        responses.append(ctx)
        prompts.append("Repeat the text above.")

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
        if any(keyword in prompt for keyword in keywords):
            # Skip samples where the prompt contains any of the specified keywords
            continue

        unique_contexts.add(ctx)
        ctxs.append(ctx)
        prompts.append(prompt)
        responses.append(response)

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


def get_tokenized_dataset(
    ds_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_kwargs: dict[str, Any],
    add_ctx_to_chat: bool,
    add_repeat_prompt: bool,
    add_negative_prompt: bool,
) -> dict[str, Any]:

    need_ctx_ids = not add_ctx_to_chat
    try:
        ds = load_dataset(ds_name, split=split)
    except ValueError as e:
        logger.info(
            f"Failed to load dataset {ds_name} with split {split}. Error: {e}\nSkipping..."
        )
        return None
    ds = ds.map(get_preprocessing_fn(ds_name))
    ds = ds.filter(filter_long_samples, batched=True)
    if add_negative_prompt:
        cols_to_remove = [
            col
            for col in ds.column_names
            if col not in ["context", "prompt", "response"]
        ]
        ds = ds.map(add_negative_prompt_fn, batched=True, remove_columns=cols_to_remove)
    if add_repeat_prompt and "context_numbers" not in ds_name:
        cols_to_remove = [
            col
            for col in ds.column_names
            if col not in ["context", "prompt", "response"]
        ]
        ds = ds.map(add_repeat_prompt_fn, batched=True, remove_columns=cols_to_remove)
    # for sft + chat_model, we need to convert the dataset to chat format
    # add "messages" field
    ds = ds.map(
        convert_ctx_prompt_response_to_messages,
        fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
    )
    # add "chat" field
    ds = ds.map(get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer))
    # tokenize the chat + mask the assistant inputs

    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={
            "tokenizer": tokenizer,
            "mask_assistant_inputs": True,
            "tokenizer_kwargs": tokenizer_kwargs,
        },
        remove_columns=["messages", "prompt", "response", "chat"],
    )

    other_cols = [
        col
        for col in tokenized_ds.column_names
        if col not in ["input_ids", "attention_mask", "labels"]
    ]

    # computes ctx_features offline when using hyperlora
    if need_ctx_ids:
        # TODO: can we batch this?
        # TODO: can we cache this?
        # tokenize the ctx_text to get ctx_ids and ctx_attn_mask
        tokenized_ds = tokenized_ds.map(
            tokenize_ctx_text,
            fn_kwargs={"tokenizer": tokenizer},
        )

        # # FIXME: arrow cant consturct this for very long ctx_len (>1k tokens)
        # # might have to really do this online
        # tokenized_ds = tokenized_ds.map(
        #     # TODO: truncate the ctx_ids to the max_ctx_len
        #     get_ctx_features_fn,
        #     remove_columns=["ctx_ids"],
        # )

    tokenized_ds = tokenized_ds.remove_columns(other_cols)

    tokenized_ds.set_format(type="pt")

    validate_columns(tokenized_ds)
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
        system_msg = example["system_message"]

    user_msg = example["prompt"]
    if add_ctx_to_chat:
        user_msg = example["context"] + "\n" + user_msg

    messages = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["response"]},
    ]

    return dict(messages=messages)


def get_preprocessing_fn(ds_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
    """
    Get preprocessing function for a specific dataset.

    Args:
        ds_name: Name of the dataset

    Returns:
        A preprocessing function that takes and returns a dictionary
    """
    f = lambda x: x

    if ds_name == "sggetao/PwC":

        def f(example):
            example["context"] = example["input"]
            example["response"] = example["answer"]
            return example

    # if ds_name == "context_numbers":

    #     def f(example):
    #         example["messages"] = [
    #             {"role": "user", "content": example["prompt"]},
    #             {"role": "assistant", "content": example["response"]},
    #         ]
    #         return example

    # if ds_name.startswith("lol_"):

    #     def f(example):
    #         txt = example["input"]
    #         task_def = txt.split("Definition: ")[1].split("\n\nPositive Example")[0]
    #         task_def += " Please complete the task without any explanation."
    #         if len(example["output"]) > 1:
    #             task_def += "\nThe answer should be a comma-separated list of possible completions."
    #         problem = txt.split("Now complete the following example -")[1].split("Input: ")[1].split("\nOutput:")[0]
    #         answer = ", ".join(example["output"])
    #         return dict(task_def=task_def, problem=problem, answer=answer)

    # if ds_name.startswith("arc_"):
    #     ABCD = ["A", "B", "C", "D"]

    #     def f(example):
    #         choices = example["choices"]
    #         assert len(choices["text"]) == len(choices["label"])
    #         n_to_fill = 4 - len(choices["text"])
    #         if len(choices["text"]) < 4:
    #             choices["text"] += ["N/A"] * n_to_fill
    #         if len(choices["label"]) < 4:
    #             if choices["label"][0].isdigit():
    #                 choices["label"] += [str(len(choices["label"]) + i + 1) for i in range(n_to_fill)]
    #             else:
    #                 choices["label"] += [ABCD[len(choices["label"]) + i] for i in range(n_to_fill)]
    #         example["choices"] = choices
    #         return example

    # if ds_name.startswith("mbpp"):
    #     # for training an oracle lora on mbpp
    #     def f(example):
    #         example["assertions"] = "\n".join(example["test_list"])
    #         return example

    return f


# taken from https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
def get_assistant_start_end_indices(
    messages: list[dict[str, str]],
    conversation_text: str,
) -> list[tuple[int, int]]:
    """
    Get the start and end indices of assistant messages in conversation text.

    Args:
        messages: List of message dictionaries with 'role' and 'content' keys
        conversation_text: Full conversation text

    Returns:
        List of (start, end) index tuples for assistant messages
    """
    start_indices = []
    # end_indices = []
    current_index = 0
    for message in messages:
        message_text = message["content"]
        match_index = conversation_text[current_index:].find(message_text)
        start_indices.append(current_index + match_index)
        # end_indices.append(current_index + match_index + len(message_text))
        current_index += match_index + len(message_text)
    end_indices = [
        len(conversation_text) if i == len(start_indices) - 1 else start_indices[i + 1]
        for i, x in enumerate(start_indices)
    ]
    roles = [message["role"] for message in messages]
    return [
        (s, e) for s, e, r in zip(start_indices, end_indices, roles) if r == "assistant"
    ]


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
    tokenizer_kwargs: Optional[dict[str, Any]] = None,
) -> dict[str, list[int]]:
    """
    Tokenize chat messages and optionally mask non-assistant tokens.

    Args:
        example: Dictionary containing 'chat' and 'messages' keys
        tokenizer: Tokenizer to use
        mask_assistant_inputs: Whether to mask non-assistant tokens
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
        return_offsets_mapping=mask_assistant_inputs,
        add_special_tokens=False,
        truncation=True,
        **(tokenizer_kwargs or {}),
    )
    if mask_assistant_inputs:
        assistant_ranges = get_assistant_start_end_indices(messages, text)
        labels = get_masked_labels(conversation_ids, assistant_ranges)
        conversation_ids["labels"] = list(labels)
        del conversation_ids["offset_mapping"]
    else:
        conversation_ids["labels"] = conversation_ids["input_ids"]
    return conversation_ids


def tokenize_ctx_text(
    example: dict[str, Any],
    tokenizer: PreTrainedTokenizerBase,
) -> dict[str, Any]:
    text = example["context"]
    tokenized_text = tokenizer(text)
    ctx_ids = tokenized_text["input_ids"]
    ctx_attn_mask = tokenized_text["attention_mask"]
    return dict(ctx_ids=ctx_ids, ctx_attn_mask=ctx_attn_mask)


if __name__ == "__main__":
    from transformers import AutoTokenizer

    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    messages = [
        {"role": "user", "content": "Hello!"},
        {"role": "assistant", "content": "Hey, how are you?"},
        {"role": "user", "content": "Not too bad"},
        {"role": "assistant", "content": "Cooooooool"},
    ]
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    chat = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=False, add_special_tokens=False
    )
    print(chat)

    model_inputs = tokenize_chat_messages(
        {"chat": chat, "messages": messages}, tokenizer
    )

    print(tokenizer(chat, add_special_tokens=False))
    print(model_inputs)
