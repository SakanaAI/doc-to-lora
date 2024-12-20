import pandas as pd
from typing import Literal
from transformers import PreTrainedTokenizerBase

from training_utils import TRAINING_TASK

IGNORE_INDEX = -100


def get_sft_prompt_formatting_fn(
    sft_mode: TRAINING_TASK,
    tokenizer: PreTrainedTokenizerBase,
):
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
        out = dict()
        out["context"] = example["context"]

        chat_text = tokenizer.apply_chat_template(
            example["messages"], tokenize=False, add_generation_prompt=False
        )
        out["messages"] = example["messages"]
        out["chat"] = chat_text
        return out

    # return f if not apply_chat_template_fn is not None else f_intx
    return f_intx


def convert_ctx_prompt_response_to_messages(example, add_ctx_to_chat=True):
    if "prompt" not in example or "response" not in example:
        raise ValueError(f"'prompt' and 'response' are required. Got: {example}")

    system_msg = ""
    if "system_message" in example:
        system_msg = example["system_message"]

    user_msg = example["prompt"]
    if add_ctx_to_chat:
        user_msg = example["context"] + "\n" + user_msg

    example["messages"] = [
        {"role": "system", "content": system_msg},
        {"role": "user", "content": user_msg},
        {"role": "assistant", "content": example["response"]},
    ]

    return example


def get_preprocessing_fn(ds_name):
    f = lambda x: x

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


# def get_inp_tokenize_fn(
#     tokenizer,
#     sft_mode: Literal["causal_lm", "completion"],
#     is_intx_model: bool,
#     inp_max_len: int,
# ):
#     def tokenize_causal_lm(examples):
#         # a dict with keys: ["input_ids", "attention_mask"]
#         tokenized_seq = tokenizer(
#             examples["text"],
#             # apply_chat_template should already add all the special tokens
#             add_special_tokens=True if not is_intx_model else False,
#             truncation=True,
#             padding=False,
#             max_length=inp_max_len,
#         )
#         tokenized_seq["labels"] = tokenized_seq["input_ids"]
#         return tokenized_seq

#     # NOTE: we're not considering multi-turn sft
#     # this fn is used to mask out the loss from the prompt
#     # and train only on the response
#     # see # see https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
#     # https://github.com/huggingface/notebooks/blob/main/examples/question_answering.ipynb
#     # for more advanced multi-turn training
#     def tokenize_prompt_completion(examples):
#         # a dict with keys: ["input_ids", "attention_mask"]
#         # we can also access seqeunce_ids to differentiate between prompt and response
#         tokenized_seq = tokenizer(
#             examples["prompt"],
#             examples["response"],
#             add_special_tokens=False,
#             truncation=True,
#             padding=False,
#             max_length=inp_max_len,
#         )

#         tokenized_seq["labels"] = [None] * len(tokenized_seq["input_ids"])
#         input_ids = tokenized_seq["input_ids"]
#         attention_mask = tokenized_seq["attention_mask"]
#         labels = tokenized_seq["labels"]
#         for i in range(len(tokenized_seq["input_ids"])):
#             if not is_intx_model:
#                 # manually add bos and eos tokens
#                 input_ids[i] = (
#                     [tokenizer.bos_token_id] + input_ids[i] + [tokenizer.eos_token_id]
#                 )
#                 attention_mask[i] = [1] + attention_mask[i] + [1]
#                 sequence_ids = [0] + tokenized_seq.sequence_ids(i) + [1]
#             else:
#                 sequence_ids = tokenized_seq.sequence_ids(i)
#             labels[i] = [
#                 -100 if sequence_id == 0 else label
#                 for sequence_id, label in zip(sequence_ids, input_ids[i])
#             ]
#         return tokenized_seq

#     tokenize_function = (
#         tokenize_causal_lm if sft_mode == "causal_lm" else tokenize_prompt_completion
#     )
#     return tokenize_function


# taken from https://github.com/huggingface/trl/issues/632#issuecomment-1972630547
def get_assistant_start_end_indices(messages, conversation_text):
    """
    Get the start and end indices of the assistant's messages in the conversation text.
    """
    start_indices = []
    current_index = 0
    for message in messages:
        message_text = message["content"]
        match_index = conversation_text[current_index:].find(message_text)
        start_indices.append(current_index + match_index)
        current_index += match_index + len(message_text)
    end_indices = [
        len(conversation_text) if i == len(start_indices) - 1 else start_indices[i + 1]
        for i, x in enumerate(start_indices)
    ]
    roles = [message["role"] for message in messages]
    return [
        (s, e) for s, e, r in zip(start_indices, end_indices, roles) if r == "assistant"
    ]


def get_masked_labels(conversation_ids, assistant_ranges):
    for id_, (id_s, id_e) in list(
        zip(conversation_ids["input_ids"], conversation_ids["offset_mapping"])
    ):
        if any(id_s >= s and id_e <= e for s, e in assistant_ranges):
            yield id_
        else:
            yield IGNORE_INDEX


def tokenize_chat_messages(
    example,
    tokenizer,
    mask_assistant_inputs=True,
    tokenizer_kwargs=None,
):
    # should be used only with chat models
    text = example["chat"]
    messages = example["messages"]
    conversation_ids = tokenizer(
        text,
        return_offsets_mapping=mask_assistant_inputs,
        **tokenizer_kwargs if tokenizer_kwargs is not None else {},
    )
    if mask_assistant_inputs:
        assistant_ranges = get_assistant_start_end_indices(messages, text)
        labels = get_masked_labels(conversation_ids, assistant_ranges)
        conversation_ids["labels"] = list(labels)
        del conversation_ids["offset_mapping"]
    else:
        conversation_ids["labels"] = conversation_ids["input_ids"]
    return conversation_ids


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
