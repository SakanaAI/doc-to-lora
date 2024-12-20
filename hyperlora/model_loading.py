import logging
import os

import torch
from peft import PeftModel, LoraConfig, VeraConfig, PeftConfig
from peft import get_peft_config as _get_peft_config
from peft.utils import PeftType
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoModel

from hyper_llm_modulator.utils.pooling import get_pooling_fn
from hyper_llm_modulator.utils.preprocessing import add_full_stop, apply_sfr_template

logger = logging.getLogger()


def get_model_and_tokenizer(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    tokenizer_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    model = get_model(
        model_name_or_path,
        train,
        requires_grad,
        use_flash_attn,
        peft_config,
        model_kwargs,
        device,
        dtype,
    )
    tokenizer = get_tokenizer(model_name_or_path, tokenizer_kwargs, peft_config, train)
    return model, tokenizer


def get_tokenizer(
    model_name_or_path, tokenizer_kwargs=None, peft_config=None, train=False
):
    # tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, padding_side="left")
    # NOTE: lora models don't have tokenizer config in the folder

    # left pad for generation, right pad for training (why?)
    padding_side = "left" if not train else "right"
    truncation_side = "left"

    # tokenizer = AutoTokenizer.from_pretrained(
    #         "models/Mistral-7B-v0.1/", padding_side=padding_side
    #     )
    if peft_config:
        model_name_or_path = peft_config.base_model_name_or_path

    if tokenizer_kwargs is None:
        tokenizer_kwargs = {}

    tokenizer = AutoTokenizer.from_pretrained(
        model_name_or_path,
        padding_side=padding_side,
        truncation_side=truncation_side,
        **tokenizer_kwargs,
    )

    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
        # tokenizer.pad_token = tokenizer.eos_token

    # tokenizer.pad_token = tokenizer.eos_token
    # tokenizer.add_bos_token = True

    # assert os.path.exists(f"{model_name_or_path}/chat_template.jinja"), (
    #     f"Chat template not found in {model_name_or_path}\n\n"
    #     "We assume a specfic form of chat template for consistency between models. Please use the templates provided",
    # )
    # if os.path.exists(f"{model_name_or_path}/chat_template.jinja"):
    #     chat_template = open(f"{model_name_or_path}/chat_template.jinja").read()
    #     chat_template = chat_template.replace("    ", "").replace("\n", "")
    #     tokenizer.chat_template = chat_template

    # tokenizer.add_eos_token = False
    # if train:
    #     # NOTE: this correctly add an eos token that has attention = 1
    #     # while padding eos tokens have attention = 0
    #     tokenizer.add_eos_token = True

    #     # shouldn't be needed as we manually compute the labels now
    #     # NOTE: this seems oddly important
    #     # and is not necessarily in the alignment handbook scripts
    #     # tokenizer.pad_token = tokenizer.unk_token  # fix model not generating eos

    return tokenizer


def get_model(
    model_name_or_path,
    train,
    requires_grad,
    use_flash_attn=True,
    peft_config=None,
    model_kwargs=None,
    device="cuda:0",
    dtype=torch.bfloat16,
):
    # TODO: we could use lower precision during finetuning
    model_init_kwargs = dict(
        pretrained_model_name_or_path=model_name_or_path,
        device_map=device,
        torch_dtype=dtype,
        trust_remote_code=True,
        # attn_implementation="flash_attention_2",
        # load_in_4bit=True,
        # load_in_8bit=True,
    )
    if model_kwargs is not None:
        model_init_kwargs.update(model_kwargs)
    if use_flash_attn:
        model_init_kwargs["attn_implementation"] = "flash_attention_2"
        # for training disable cache
    if train:
        model_init_kwargs["use_cache"] = False
    logger.debug(f"Model init kwargs: {model_init_kwargs}")
    model = AutoModelForCausalLM.from_pretrained(**model_init_kwargs)
    if peft_config is not None:
        model = PeftModel(model, peft_config)
    model.train(train)
    for param in model.parameters():
        param.requires_grad = requires_grad
    return model


def get_lora_config(model_dir, **kwargs):
    r = kwargs.get("r", 8)
    peft_conf_kwargs = dict(
        r=r,
        peft_type=PeftType.LORA,
        base_model_name_or_path=model_dir,
        task_type="CAUSAL_LM",
        lora_dropout=0.05,
        lora_alpha=r ** (3 / 2) * 2,
    )

    peft_conf_kwargs.update(kwargs)
    peft_config = _get_peft_config(peft_conf_kwargs)
    return peft_config


# def get_emb_model_and_fns(emb_model_name, device):
#     emb_model = AutoModel.from_pretrained(
#         emb_model_name,
#         device_map=device,
#         torch_dtype=torch.float32 if "gte" in emb_model_name else torch.bfloat16,
#         trust_remote_code=True,
#     ).eval()
#     emb_tokenizer = AutoTokenizer.from_pretrained(emb_model_name)
#     if emb_tokenizer.pad_token_id is None:
#         emb_tokenizer.pad_token_id = emb_tokenizer.eos_token_id
#         # assert bool(args.query_intx), "query_intx must be provided for the emb_model_name"
#         # partial(get_detailed_instruct, task_description=args.query_intx)
#     task_desc_format_fn = add_full_stop
#     if "SFR" in emb_model_name:
#         task_desc_format_fn = apply_sfr_template
#         pooling_fn = get_pooling_fn("last_token")
#     elif "gte" in emb_model_name:
#         pooling_fn = get_pooling_fn("cls")
#     return emb_model, emb_tokenizer, task_desc_format_fn, pooling_fn
