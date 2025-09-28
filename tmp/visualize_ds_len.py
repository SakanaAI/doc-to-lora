import os
from functools import partial

import matplotlib.pyplot as plt
import numpy as np

from ctx_to_lora.data.processing import get_tokenized_dataset
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.utils import check_is_iterable

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
tokenized_ds = dict()
val_ds_names = []  # ["fw_qa_large", "ctx_qa", "pwc", "hotpot_qa", "squad"]
for split, ds_names in zip(["train", "validation"], [train_ds_names, val_ds_names]):
    tokenized_ds[split] = {
        os.path.basename(ds_name): _get_tokenized_dataset(ds_name, split)
        for ds_name in ds_names
    }
    for ds_name in tokenized_ds[split]:
        print(ds_name)
        print(f"Num samples: {len(tokenized_ds[split][ds_name])}")
        sample = tokenized_ds[split][ds_name][0]

        # TODO: let's investigate samples that are very long and very short
        if check_is_iterable(sample["input_ids"]):
            tokens = []
            for sample in tokenized_ds[split][ds_name]:
                tokens += [len(ids) for ids in sample["input_ids"]]
        else:
            tokens = [
                len(sample["input_ids"]) for sample in tokenized_ds[split][ds_name]
            ]

        print(f"Num avg. tokens: {sum(tokens) / len(tokens)}")
        print(f"Num max. tokens: {max(tokens)}")
        print(f"Num min. tokens: {min(tokens)}")
        print(f"Num std. tokens: {np.std(tokens)}")
        plt.hist(tokens, bins=100)
        plt.savefig(f"tmp/n_token_hist_{split}_{ds_name}.png")
        plt.close()

        long_indices = [i for i, n_tokens in enumerate(tokens) if n_tokens > 100]
        short_indices = [i for i, n_tokens in enumerate(tokens) if n_tokens < 30]
        breakpoint()
        for i in long_indices:
            print(f"Long sample {i}: {tokenized_ds[split][ds_name][i]}")
        breakpoint()
        for i in short_indices:
            print(f"Short sample {i}: {tokenized_ds[split][ds_name][i]}")

        ctx_tokens = [len(sample["ctx_ids"]) for sample in tokenized_ds[split][ds_name]]
        print(f"Num avg. ctx tokens: {sum(ctx_tokens) / len(ctx_tokens)}")
        print(f"Num max. ctx tokens: {max(ctx_tokens)}")
        print(f"Num min. ctx tokens: {min(ctx_tokens)}")
        print(f"Num std. ctx tokens: {np.std(ctx_tokens)}")
        plt.hist(ctx_tokens, bins=100)
        plt.savefig(f"tmp/ctx_n_token_hist_{split}_{ds_name}.png")
        plt.close()
