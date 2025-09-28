import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ctx_to_lora.data.processing import get_tokenized_dataset, load_and_process_dataset
from ctx_to_lora.model_loading import get_tokenizer

tokenizer_kwargs = {"max_length": 2**13}
ctx_tokenizer_kwargs = {"max_length": 2**13}  # not used for now
tokenizer = get_tokenizer("google/gemma-2-2b-it", train=True)
ctx_tokenizer = get_tokenizer("google/gemma-2-2b-it", train=True)

_get_tokenized_dataset = partial(
    get_tokenized_dataset,
    base_model_max_len=2**13,
    tokenizer=tokenizer,
    tokenizer_kwargs=tokenizer_kwargs,
    ctx_model_max_len=2**13,
    ctx_tokenizer=ctx_tokenizer,
    ctx_tokenizer_kwargs=ctx_tokenizer_kwargs,
    add_ctx_to_chat=False,
    add_repeat_prompt=False,
    add_negative_prompt=False,
    use_kl_loss=False,
    # streaming=data_args.streaming,
)
_load_and_process_dataset = partial(
    load_and_process_dataset,
    add_negative_prompt=False,
    add_repeat_prompt=False,
    repeat_prob=0,
    is_pretrain=False,
    streaming=False,
    num_proc=8,
)

train_ds_names = [
    # "fw_qa",
    # "fw_qa_large",
    # "ctx_qa",
    # "pwc",
    # "hotpot_qa",
    # "squad",
    # "drop",
    # "narrativeqa",
    # "quoref",
    # "ropes",
    # "synthetic_convqa",
    # "booksum",
    # "gov_report",
    # "wikitext-2",
    # "wikitext-103",
    # "fw_qa_2"
    # "fw_qa_xl",
    # "fw_qa_3_medium",
    "self_gen/google/gemma-2-2b-it_temp_0.0/fw_qa_v2_2k_len_level_0_tiny",
    "self_gen/google/gemma-2-2b-it_temp_0.0/fw_qa_v2_2k_len_level_1_tiny",
    "self_gen/google/gemma-2-2b-it_temp_0.0/fw_qa_v2_2k_len_level_2_tiny",
    "self_gen/google/gemma-2-2b-it_temp_0.0/fw_qa_v2_2k_len_level_3_tiny",
]
loaded_dataset = dict()
val_ds_names = []  # ["fw_qa_large", "ctx_qa", "pwc", "hotpot_qa", "squad"]
for split, ds_names in zip(["train", "validation"], [train_ds_names, val_ds_names]):
    loaded_dataset[split] = {
        os.path.basename(ds_name): _load_and_process_dataset(ds_name, split)
        for ds_name in ds_names
    }
    for ds_name in loaded_dataset[split]:
        print(ds_name)
        print(f"Num samples: {len(loaded_dataset[split][ds_name])}")
        ds = loaded_dataset[split][ds_name]
        sample = loaded_dataset[split][ds_name][0]

        # TODO: let's investigate samples that are very long and very short
        if "responses" in sample:
            char_lens = []
            responses = []
            questions = []
            ctxs = []
            for sample in loaded_dataset[split][ds_name]:
                char_lens += [len(res) for res in sample["responses"]]
                responses += sample["responses"]
                questions += sample["prompts"]
                ctxs += [sample["context"]] * len(sample["responses"])
        else:
            responses = ds["response"]
            questions = ds["prompt"]
            ctxs = ds["context"]

            char_lens = [
                len(sample["response"]) for sample in loaded_dataset[split][ds_name]
            ]

        print(f"Num avg. char_lens: {sum(char_lens) / len(char_lens)}")
        print(f"Num max. char_lens: {max(char_lens)}")
        print(f"Num min. char_lens: {min(char_lens)}")
        print(f"Num std. char_lens: {np.std(char_lens)}")
        plt.hist(char_lens, bins=100)
        plt.savefig(f"tmp/n_token_hist_{split}_{ds_name}.png")
        plt.close()

        long_indices = [i for i, clen in enumerate(char_lens) if clen > 2000]
        short_indices = [i for i, clen in enumerate(char_lens) if clen < 100]
        breakpoint()
        for i in long_indices:
            print(f"Long sample {i}")
            print(f"Response length: {len(responses[i])}")
            print(f"Context: {ctxs[i]}")
            print(f"Question: {questions[i]}")
            print(f"Response: {responses[i]}")
            print("=" * 80)

        breakpoint()
        for i in short_indices:
            print(f"Short sample {i}")
            print(f"Response length: {len(responses[i])}")
            print(f"Context: {ctxs[i]}")
            print(f"Question: {questions[i]}")
            print(f"Response: {responses[i]}")
            print("=" * 80)

        ctx_tokens = [
            len(sample["ctx_ids"]) for sample in loaded_dataset[split][ds_name]
        ]
        print(f"Num avg. ctx char_lens: {sum(ctx_tokens) / len(ctx_tokens)}")
        print(f"Num max. ctx char_lens: {max(ctx_tokens)}")
        print(f"Num min. ctx char_lens: {min(ctx_tokens)}")
        print(f"Num std. ctx char_lens: {np.std(ctx_tokens)}")
        plt.hist(ctx_tokens, bins=100)
        plt.savefig(f"tmp/ctx_n_token_hist_{split}_{ds_name}.png")
        plt.close()
