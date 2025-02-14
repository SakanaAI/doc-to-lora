import logging
import numpy as np
from glob import glob
from typing import Any, Callable, Iterator, Optional

from datasets import load_dataset, IterableDataset
from transformers import PreTrainedTokenizerBase

from ctx_to_lora.training_utils import TRAINING_TASK

IGNORE_INDEX = -100

logger = logging.getLogger()
FW_QA_PATHS = [
    f"data/raw_datasets/fw_qa/{i:05d}.parquet" for i in [0, 1, 6, 7, 8, 10, 22, 30, 35]
]
DS_KWARGS = {
    "hotpot_qa": dict(
        train=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[900:]"),
        validation=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[:900]"),
        test=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="validation"),
    ),
    "hotpot_qa_tiny": dict(
        train=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[900:2000]"),
        validation=dict(path="hotpotqa/hotpot_qa", name="fullwiki", split="train[:900]"),
    ),
    "pwc": dict(
        train=dict(path="sggetao/PwC", split="train[900:]"),
        validation=dict(path="sggetao/PwC", split="train[:900]"),
        test=dict(path="sggetao/PwC", split="test"),
    ),
    "pwc_tiny": dict(
        train=dict(path="sggetao/PwC", split="train[900:2000]"),
        validation=dict(path="sggetao/PwC", split="train[:900]"),
    ),
    "squad": dict(
        train=dict(path="rajpurkar/squad", split="train[900:]"),
        validation=dict(path="rajpurkar/squad", split="train[:900]"),
    ),
    "fw_qa_tiny": dict(
        train=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa/00000.parquet",
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa/00000_val.parquet",
            split="train",
        ),
    ),
    "fw_qa": dict(
        train=dict(
            path="parquet",
            data_files=FW_QA_PATHS,
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files="data/raw_datasets/fw_qa/*_val.parquet",
            split="train",
        ),
    ),
    "fw_qa_large": dict(
        train=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_large/*res.parquet"),
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_large/*val.parquet"),
            split="train",
        ),
    ),
    "fw_qa_xl": dict(
        train=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_xl/*[!val].parquet"),
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/fw_qa_xl/*val.parquet"),
            split="train",
        ),
    ),
    "ctx_qa": dict(
        train=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/ctx_qa/*res.parquet"),
            split="train",
        ),
        validation=dict(
            path="parquet",
            data_files=glob("data/raw_datasets/ctx_qa/*val.parquet"),
            split="train",
        ),
    ),
    "drop": dict(
        train=dict(
            path="ucinlp/drop",
            split="train",
        ),
    ),
    "narrativeqa": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="narrativeqa",
            split="train",
        ),
    ),
    "quoref": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="quoref",
            split="train",
        ),
    ),
    "ropes": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="ropes",
            split="train",
        ),
    ),
    "synthetic_convqa": dict(
        train=dict(
            path="nvidia/ChatQA-Training-Data",
            name="synthetic_convqa",
            split="train",
        ),
    ),
    "gsm8k": dict(
        train=dict(
            path="openai/gsm8k",
            name="main",
            split="train[100:]",
        ),
        validation=dict(
            path="openai/gsm8k",
            name="main",
            split="train[:100]",
        ),
    ),
    "openmathintx-2": dict(
        train=dict(
            path="nvidia/OpenMathInstruct-2",
            split="train_1M[:100000]",
        ),
    ),
    "opencoder-edu": dict(
        train=dict(
            path="OpenCoder-LLM/opc-sft-stage2",
            name="educational_instruct",
            split="train[900:]",
        ),
        validation=dict(
            path="OpenCoder-LLM/opc-sft-stage2",
            name="educational_instruct",
            split="train[:900]",
        ),
    ),
}


def get_ds_kwargs(ds_name: str, split: str) -> dict[str, Any]:
    if (ds_name not in DS_KWARGS) or (split not in DS_KWARGS[ds_name]):
        kwargs = dict(path=ds_name, split=split)
        logger.warning(
            f"No dataset kwargs found for '{ds_name}' with split '{split}'.\n"
            f"Using default kwargs: {kwargs}"
        )
        return kwargs
    return DS_KWARGS[ds_name][split]


def validate_columns(tokenized_ds):
    cols = ["input_ids", "attention_mask", "labels"]
    if "ctx_ids" in tokenized_ds.column_names:
        cols += ["ctx_ids", "ctx_attn_mask"]
    if "chat_ids" in tokenized_ds.column_names:
        cols += ["chat_ids", "chat_attn_mask", "chat_labels"]
    ref_cols = set(cols)
    assert (
        set(tokenized_ds.column_names) == ref_cols
    ), f"Columns mismatch: {set(tokenized_ds.column_names)} != {ref_cols}"


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
        unique_contexts.add(ctx)
        # remove too long contexts
        if len(ctx) > 3000:
            continue
        ctxs.append(ctx)
        responses.append(ctx)
        prompts.append("Repeat the text above.")

    logger.debug(f"Adding repeat prompt...")
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


def get_tokenized_dataset(
    ds_name: str,
    split: str,
    tokenizer: PreTrainedTokenizerBase,
    tokenizer_kwargs: dict[str, Any],
    ctx_tokenizer: PreTrainedTokenizerBase,
    ctx_tokenizer_kwargs: dict[str, Any],
    add_ctx_to_chat: bool,
    add_repeat_prompt: bool,
    add_negative_prompt: bool,
    use_kl_loss: bool,
    set_format: Optional[str] = None,
) -> dict[str, Any]:

    logger.debug(f"Loading dataset {ds_name} with split {split}...")
    need_ctx_ids = not add_ctx_to_chat
    try:
        ds = load_dataset(**get_ds_kwargs(ds_name, split), trust_remote_code=True)
    except ValueError as e:
        logger.info(
            f"Failed to load dataset {ds_name} with split {split}. Error: {e}\nSkipping..."
        )
        return None
    cols_to_remove = [
        col for col in ds.column_names if col not in ["context", "prompt", "response"]
    ]
    ds = ds.map(get_preprocessing_fn(ds_name), num_proc=16)
    ds = ds.remove_columns(cols_to_remove)
    ds = ds.filter(filter_none, batched=True, num_proc=16)
    ds = ds.filter(filter_long_samples, batched=True, num_proc=16)
    if split == "train":
        if add_negative_prompt:
            ds = ds.map(add_negative_prompt_fn, batched=True, batch_size=None)
        if add_repeat_prompt and "context_numbers" not in ds_name:
            ds = ds.map(add_repeat_prompt_fn, batched=True, batch_size=None)
    tokenized_ds = construct_and_tokenize_ctx_qa(
        tokenizer,
        tokenizer_kwargs,
        ctx_tokenizer,
        ctx_tokenizer_kwargs,
        add_ctx_to_chat,
        use_kl_loss,
        need_ctx_ids,
        ds,
        set_format,
    )
    return tokenized_ds


def construct_and_tokenize_ctx_qa(
    tokenizer,
    tokenizer_kwargs,
    ctx_tokenizer,
    ctx_tokenizer_kwargs,
    add_ctx_to_chat,
    use_kl_loss,
    need_ctx_ids,
    ds,
    set_format=None,
):
    # for sft + chat_model, we need to convert the dataset to chat format
    # add "messages" field
    ds = ds.map(
        convert_ctx_prompt_response_to_messages,
        fn_kwargs={"add_ctx_to_chat": add_ctx_to_chat},
        num_proc=16,
    )
    # add "chat" field
    ds = ds.map(
        get_sft_prompt_formatting_fn(TRAINING_TASK.COMPLETION, tokenizer), num_proc=16
    )
    # tokenize the chat + mask the assistant inputs
    ds = ds.filter(filter_long_chat, batched=True, num_proc=16)

    # add "input_ids", "attention_mask", "labels"
    tokenized_ds = ds.map(
        tokenize_chat_messages,
        fn_kwargs={
            "tokenizer": tokenizer,
            "mask_assistant_inputs": True,
            "tokenizer_kwargs": tokenizer_kwargs,
        },
        num_proc=16,
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
            num_proc=16,
        )

    if need_ctx_ids:
        # TODO: can we batch this?
        # TODO: can we cache this?
        # tokenize the ctx_text to get ctx_ids and ctx_attn_mask
        tokenized_ds = tokenized_ds.map(
            tokenize_ctx_text,
            fn_kwargs={"tokenizer": ctx_tokenizer},
            batched=True,
            num_proc=16,
        )

    tokenized_ds = tokenized_ds.remove_columns(
        ["messages", "chat", "context", "prompt", "response"]
    )
    tokenized_ds.set_format(type=set_format)
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


def get_preprocessing_fn(ds_name: str) -> Callable[[dict[str, Any]], dict[str, Any]]:
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

            return {
                "context": txt.strip(),
                "prompt": sample["question"],
                "response": sample["answer"],
            }

    elif "squad" in ds_name:

        def f(sample):
            return {
                "context": sample["context"],
                "prompt": sample["question"],
                "response": sample["answers"]["text"][0],
            }

    elif ds_name == "drop":

        def f(sample):
            return {
                "context": sample["passage"],
                "prompt": sample["question"],
                "response": ", ".join(sample["answers_spans"]["spans"]),
            }

    elif ds_name in ["narrativeqa", "quoref", "ropes", "synthetic_convqa"]:

        def f(sample):
            response = sample["answers"][0]
            if isinstance(response, list):
                response = response[0]
            return {
                "context": sample["document"],
                "prompt": sample["messages"][0]["content"],
                "response": response,
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

    return f


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
