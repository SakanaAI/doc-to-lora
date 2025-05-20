import logging
import os
import random
import string
import time
from multiprocess import set_start_method
from math import ceil
from collections import defaultdict
from copy import copy, deepcopy
from functools import partial
from importlib.resources import read_binary
from typing import Callable, Optional

import numpy as np
import torch
import wandb
import yaml

from ctx_to_lora.data_utils import (
    DS_LEN,
    convert_ctx_prompt_response_to_messages,
    get_preprocessing_fn,
    get_sft_prompt_formatting_fn,
    get_tokenized_dataset,
    tokenize_chat_messages,
    tokenize_ctx_text,
)
from datasets import (
    concatenate_datasets,
    interleave_datasets,
    disable_caching,
    load_dataset,
    IterableDataset,
)
from ctx_to_lora.model_loading import (
    get_lora_config,
    get_model_and_tokenizer,
    get_tokenizer,
)
from ctx_to_lora.modeling_utils import (
    EarlyExit,
    HyperLoRA,
    ModulatedPretrainedModel,
    get_hypernet_config,
)
from rouge_score import rouge_scorer
from ctx_to_lora.training_utils import TRAINING_TASK, train_model
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    AutoConfig,
    PretrainedConfig,
    DataCollatorForSeq2Seq,
    EvalPrediction,
    HfArgumentParser,
    set_seed,
)
from accelerate import Accelerator, DeepSpeedPlugin
from peft import PeftModel
from transformers.utils import is_liger_kernel_available
from transformers.data import DataCollatorWithFlattening
from torch.utils.data import DataLoader
from ctx_to_lora.utils import (
    extract_cli_args,
    get_base_model,
    get_run_name,
    log_num_train_params,
    save_yaml,
    setup_logging,
    validate_args,
)

from ctx_to_lora.configs import (
    ArgumentParser,
    CtxTrainingArguments,
    TrainingArguments,
    DataArguments,
    ExperimentSetup,
    LoRAArguments,
    ModelArguments,
    HypernetArguments,
    AggregatorArguments,
    CtxEncoderArguments,
)

logger = logging.getLogger()

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def compute_per_token_acc(shift_logits, shift_labels, valid_masks):
    indices = torch.where(valid_masks)
    # acc = (shift_logits.argmax(-1) == shift_labels)[indices].float().mean().item()
    # return {"per_token_acc": acc}
    acc = (shift_logits.argmax(-1) == shift_labels)[indices].float()
    return {
        "per_token_accs": acc.flatten().tolist(),
        "n_per_token_accs": valid_masks.sum().item(),
    }


def compute_prefix_matching(shift_logits, shift_labels, valid_masks):
    lengths = valid_masks.sum(dim=1)

    is_wrong = (shift_logits.argmax(-1) != shift_labels) * valid_masks
    is_correct = (shift_logits.argmax(-1) == shift_labels) * valid_masks
    # NOTE: not reliable for multi-turn conversations
    # ie, all tokens in the following user's turn will be correct
    # still monotonically correlate with perf though
    wrong_pos = torch.argmax(is_wrong, dim=1) - torch.argmax(valid_masks, dim=1)
    perf = wrong_pos / lengths

    # if all tokens are correct, set to 1
    perf = torch.where(is_correct.sum(dim=1) == lengths, 1, perf)
    # return {"prefix_matching": perf.mean().item()}
    return {
        "prefix_matchings": perf.tolist(),
        "n_prefix_matchings": valid_masks.shape[0],
    }


@torch.no_grad()
def compute_perplexity(shift_logits, shift_labels, valid_masks):
    loss_fct = torch.nn.CrossEntropyLoss(reduction="none")
    loss = loss_fct(shift_logits.transpose(1, 2), shift_labels)
    loss = (loss * valid_masks).sum(dim=1) / valid_masks.sum(dim=1)
    # perplexity = torch.exp(loss).mean().item()
    # return {"perplexity": perplexity}
    preplexities = torch.exp(loss)
    return {
        "perplexities": preplexities.tolist(),
        "n_perplexities": valid_masks.shape[0],
    }


class Evaluator:
    def __init__(self, metric_fns: list[Callable]):
        self.metric_fns = metric_fns
        self.reset()

    def reset(self):
        self.accum_metrics = defaultdict(list)
        self.count = defaultdict(list)

    def update(self, shift_logits, shift_labels, valid_masks):
        for metric_fn in self.metric_fns:
            metric = metric_fn(shift_logits, shift_labels, valid_masks)
            for k, v in metric.items():
                if k.startswith("n_"):
                    self.count[k[2:]].append(v)
                else:
                    self.accum_metrics[k] += v

    def compute(self):
        # Get result across entire eval set
        result = {
            k: np.sum(v) / np.sum(self.count[k]) for k, v in self.accum_metrics.items()
        }
        # Reset batch statistics
        self.reset()
        return result


@torch.no_grad()
def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
) -> Optional[dict]:
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks)
    if compute_result:
        return evaluator.compute()


# def compute_metrics(eval_pred: EvalPrediction) -> dict:
#     """
#     Custom metrics function for the trainer
#     Args:
#         eval_pred: tuple of predictions and labels
#     Returns:
#         dictionary containing metric names (str) and values (Any)
#     """
#     # compute per token accuracy
#     # logits, labels = eval_pred.predictions, eval_pred.label_ids
#     # shift_logits = logits[..., :-1, :]
#     pred_ids, labels = eval_pred.predictions, eval_pred.label_ids
#     shift_pred_ids = pred_ids[..., :-1]
#     shift_labels = labels[..., 1:]
#     valid_masks = np.where(shift_labels != -100, 1, 0)

#     per_token_acc = compute_per_token_acc(shift_pred_ids, shift_labels, valid_masks)
#     prefix_matching = compute_prefix_matching(shift_pred_ids, shift_labels, valid_masks)
#     # entropy = compute_entropy(shift_logits, shift_labels, valid_masks)
#     return dict(
#         **per_token_acc,
#         **prefix_matching,
#         # **entropy,
#         num_valid_tokens=valid_masks.sum(),
#         num_samples=valid_masks.shape[0],
#     )


# # https://discuss.huggingface.co/t/cuda-out-of-memory-when-using-trainer-with-compute-metrics/2941/13
# def preprocess_logits_for_metrics(logits, labels):
#     """
#     Original Trainer may have a memory leak.
#     This is a workaround to avoid storing too many tensors that are not needed.
#     """
#     pred_ids = torch.argmax(logits, dim=-1)
#     return pred_ids


def main():
    ############ Argument parsing
    parser = ArgumentParser(
        (
            DataArguments,
            CtxTrainingArguments,
            ModelArguments,
            LoRAArguments,
            TrainingArguments,
            HypernetArguments,
            AggregatorArguments,
            CtxEncoderArguments,
        )
    )
    (
        data_args,
        ctx_args,
        model_args,
        lora_args,
        training_args,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    ) = parser.parse()

    # there shouldn't be overlap between args
    validate_args(
        [
            data_args,
            ctx_args,
            model_args,
            lora_args,
            training_args,
            hypernet_args,
            aggregator_args,
            ctx_encoder_args,
        ]
    )

    set_seed(training_args.seed)
    checkpoint_dir = training_args.resume_from_checkpoint
    if checkpoint_dir and not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint{checkpoint_dir} is not a directory")

    # should be the same across processes
    # still possible to have a name crash though
    # logging_dir is just "runs/DATE_TIME_HOSTNAME"
    slurm_job_id = f"_{os.getenv('SLURM_JOB_ID')}" if os.getenv("SLURM_JOB_ID") else ""
    run_name = (
        get_run_name(seed_str=training_args.logging_dir.strip("runs/") + slurm_job_id)
        if not checkpoint_dir
        else checkpoint_dir.strip("/").split("/")[-2]
    )

    output_dir = f"train_outputs/runs/{run_name}"
    setup_logging(output_dir, debug=os.getenv("DEBUG", False))
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    cli_args = extract_cli_args(os.sys.argv)
    save_yaml(cli_args, f"{output_dir}/cli_args.yaml")
    if "config" in cli_args:
        config_name = os.path.basename(cli_args["config"]).split(".yaml")[0]
        os.environ["WANDB_TAGS"] = config_name

    run_name = os.path.basename(output_dir)
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    if (
        training_args.lr_scheduler_type == "cosine_with_min_lr"
        and training_args.lr_scheduler_kwargs is None
    ):
        training_args.lr_scheduler_kwargs = {"min_lr": 1e-7}
    args = {
        **vars(deepcopy(data_args)),
        **vars(deepcopy(ctx_args)),
        **vars(deepcopy(model_args)),
        **vars(deepcopy(lora_args)),
        **vars(deepcopy(training_args)),
        **vars(deepcopy(hypernet_args)),
        **vars(deepcopy(aggregator_args)),
        **vars(deepcopy(ctx_encoder_args)),
    }
    args["deepspeed_plugin"] = None
    logger.debug(f"args: {args}")
    save_yaml(args, f"{output_dir}/args.yaml")

    ############ Model setup

    if not ctx_args.from_pretrained_checkpoint:
        model_name = model_args.model_name_or_path
        model, tokenizer = get_model_and_tokenizer(
            **vars(model_args),
            train=True,
            requires_grad=False,  # ctx_args.exp_setup == ExperimentSetup.FULL_FINETUNE,
            peft_config=get_lora_config(model_name, **vars(lora_args)),
        )
        ctx_name = ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_name is not None:
            ctx_encoder_model_config = AutoConfig.from_pretrained(
                ctx_name, trust_remote_code=True
            )
            if "Llama" in ctx_name and "Vision" in ctx_name:
                ctx_encoder_model_config = ctx_encoder_model_config.text_config
            ctx_tokenizer = get_tokenizer(ctx_name, train=True)
        else:
            ctx_name = model.base_model.config.name_or_path
            ctx_encoder_model_config = model.config
            ctx_tokenizer = tokenizer

    if ctx_args.exp_setup == ExperimentSetup.HYPERLORA:
        # TODO: handle only extra_modules case (no target_modules)
        logger.info("Using HyperLoRA")
        if not ctx_args.from_pretrained_checkpoint:
            hypernet_config = get_hypernet_config(
                model, ctx_encoder_model_config, hypernet_args, aggregator_args
            )
            if ctx_encoder_args.layer_idx is None:
                ctx_encoder_args.layer_idx = (
                    ctx_encoder_model_config.num_hidden_layers // 4
                )
                logger.info(
                    f"Using the first {ctx_encoder_args.layer_idx} layers"
                    " as the context encoder"
                )
            model = ModulatedPretrainedModel(
                model,
                hypernet_config,
                ctx_encoder_args,
                ctx_args.use_kl_loss,
            )
            training_args.gen_lora_l1_reg_coef = ctx_args.gen_lora_l1_reg_coef
        else:
            logger.info(
                f"Loading from checkpoint: {ctx_args.from_pretrained_checkpoint}"
            )
            model = ModulatedPretrainedModel.from_state_dict(
                torch.load(ctx_args.from_pretrained_checkpoint, weights_only=False),
                train=True,
                use_flash_attn=model_args.use_flash_attn,
            )
            tokenizer = get_tokenizer(model.base_model.config.name_or_path, train=True)
            ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
            if ctx_name is None:
                ctx_name = model.base_model.config.name_or_path
            ctx_tokenizer = get_tokenizer(ctx_name, train=True)
        if len([p for p in model.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in model.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")
    else:
        # activate LoRA
        base_model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        base_model_config.save_pretrained(output_dir)
        logger.info("Using LoRA")
        model.set_adapter("default")

    model.train()
    logger.debug(model)
    log_num_train_params(model)

    # model = accelerator.prepare(model)

    ############ Dataset setup

    logger.info("Loading dataset...")

    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)
    # tokenizer_kwargs = {"max_length": ctx_args.max_base_len}
    # ctx_tokenizer_kwargs = {"max_length": ctx_args.max_ctx_len}  # not used for now

    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        tokenizer_kwargs={},
        ctx_model_max_len=model.ctx_encoder.config.max_position_embeddings,
        ctx_tokenizer=ctx_tokenizer,
        ctx_tokenizer_kwargs={},
        add_ctx_to_chat=add_ctx_to_chat,
        add_repeat_prompt=ctx_args.add_repeat_prompt,
        add_negative_prompt=ctx_args.add_negative_prompt,
        use_kl_loss=ctx_args.use_kl_loss,
        set_format=None if ctx_args.use_sequence_packing else "pt",
        # streaming=data_args.streaming,
    )
    tokenized_ds = {"train": dict(), "validation": dict(), "test": dict()}
    for split, ds_names in zip(
        ["train", "validation", "test"],
        [data_args.train_ds_names, data_args.val_ds_names, data_args.test_ds_names],
    ):
        if not ds_names:
            continue
        streaming = (split == "train") and data_args.streaming
        for ds_name in ds_names:
            # with accelerator.main_process_first():
            ds = _get_tokenized_dataset(ds_name, split, streaming=streaming)
            tokenized_ds[split][os.path.basename(ds_name)] = ds

    train_ds = tokenized_ds["train"]
    logging.info(f"train_ds: {train_ds}")

    val_ds = dict()
    if "validation" in tokenized_ds:
        n_val_samples = data_args.max_val_samples_per_ds
        for ds_name, ds in tokenized_ds["validation"].items():
            if ds is None:
                # take some samples from train_ds
                ds = train_ds[ds_name].take(n_val_samples)
                train_ds[ds_name] = train_ds[ds_name].skip(n_val_samples)

            val_ds[ds_name] = ds
            val_indices = np.random.permutation(len(ds))[:n_val_samples]
            val_ds[ds_name] = val_ds[ds_name].select(val_indices)

    max_steps = ceil(
        sum([DS_LEN[ds] for ds in train_ds])
        * training_args.num_train_epochs
        / training_args.per_device_train_batch_size
        / training_args.gradient_accumulation_steps
        / training_args.world_size
    )
    training_args.max_steps = max_steps

    # interleaving streaming datasets
    # simplify the probs for smaller datasets
    # slightly upsample those datasets
    probs = [
        0.01 if "fw_qa" not in ds_name else 1 + 0.01 - 0.01 * len(train_ds)
        for ds_name in train_ds
    ]
    train_ds = interleave_datasets(
        list(train_ds.values()),
        probabilities=probs,
        stopping_strategy="all_exhausted",
        seed=training_args.seed,
    )

    logger.info(f"train_ds: {train_ds}")
    logger.info(f"val_ds: {val_ds}")
    flattener = DataCollatorWithFlattening()

    def train_packed_collator(inp_list):
        # no padding
        packed_inputs = flattener(inp_list, return_tensors="pt")
        if "ctx_ids" in inp_list[0]:
            ctx_ids = [{"input_ids": example["ctx_ids"]} for example in inp_list]
            packed_ctx = flattener(ctx_ids, return_tensors="pt")
            packed_inputs["ctx_ids"] = packed_ctx["input_ids"]
            packed_inputs["ctx_position_ids"] = packed_ctx["position_ids"]
        return packed_inputs

    def train_collator(inp_list, tokenizer):
        # input is a list of tokenized sequences
        padding_kwargs = dict(
            padding=True,
            padding_side="right",
            pad_to_multiple_of=8,
            return_tensors="pt",
        )

        ctx_ids = None
        if "ctx_ids" in inp_list[0]:
            # have to be manual since it has [ctx_len, features] shape
            # pad to the longest ctx_len in the batch
            # which can have a different length from the input_ids, attn_mask, labels

            ctx_ids = [example.pop("ctx_ids") for example in inp_list]
            ctx_ids = torch.nn.utils.rnn.pad_sequence(
                ctx_ids,
                batch_first=True,
                padding_value=0,
            )
            # exotic keys won't be padded, so we need to pad them as well
            ctx_attn_mask = [example.pop("ctx_attn_mask") for example in inp_list]
            ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
                ctx_attn_mask,
                batch_first=True,
                padding_value=0,
            )

        labels = [x.pop("labels") for x in inp_list]
        padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

        # hacky explicit padding since the labels are not padded by default
        labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
        labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
        out = {**padded_seq, "labels": labels}
        if ctx_ids is not None:
            out["ctx_ids"] = ctx_ids
            out["ctx_attn_mask"] = ctx_attn_mask

        return out

    train_collator = (
        train_packed_collator
        if ctx_args.use_sequence_packing
        else partial(train_collator, tokenizer=tokenizer)
    )

    # TODO: use SFTTrainer instead? https://huggingface.co/docs/trl/en/sft_trainer
    # TODO: use packing with SFTTrainer

    # HACK [local patch]: deepspeed model loading problem (for resume training)
    # see https://github.com/microsoft/DeepSpeed/pull/6626/files
    # /home/rujikorn_sakana_ai/.conda/envs/ctx-to-lora/lib/python3.10/site-packages/deepspeed/runtime/engine.py
    if training_args.use_liger_kernel and is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        if isinstance(model, ModulatedPretrainedModel):
            logger.info("Applying liger-kernel to ModulatedPretrainedModel")
            if isinstance(model.base_model, PeftModel):
                _apply_liger_kernel_to_instance(model=model.base_model.base_model.model)
            else:
                _apply_liger_kernel_to_instance(model=model.base_model.model)
            if ctx_name is not None:
                logger.info("Applying liger-kernel to ctx_encoder_model")
                _apply_liger_kernel_to_instance(model=model.ctx_encoder.base_model)
        elif isinstance(model, PeftModel):
            logger.info("Applying liger-kernel to PeftModel")
            _apply_liger_kernel_to_instance(model=model.base_model.model)

    if LOCAL_RANK == 0:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=run_name,
            group=run_name,
            config=args,
            tags=os.getenv("WANDB_TAGS").split(","),
            notes=ctx_args.notes,
            resume="allow",
        )
    else:
        wandb.init(mode="disabled")

    train_model(
        model,
        # tokenizer,
        training_args,
        train_ds,
        val_ds,
        train_collator,
        # partial(generation_collator, tokenizer=tokenizer),
        compute_metrics=partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_perplexity]
            ),
        ),
        # max_new_tokens=ctx_args.max_new_tokens,
        # gen_per_device_eval_batch_size=ctx_args.gen_per_device_eval_batch_size,
    )
    logger.info(f"Training run finished and saved to {output_dir}")


if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_PROJECT"] = "ctx_to_lora"
    os.environ["WANDB_WATCH"] = ""  # "all"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["KMP_AFFINITY"] = "disabled"  # fixing iterable dataset stuck
    os.environ["OMP_NUM_THREADS"] = "23"
    # os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "137438953472"  # 128 TB
    if os.getenv("DEBUG", False):
        disable_caching()
    # randomly sleep to avoid run_name collision
    # time.sleep(random.random() * 13)
    # set_start_method("spawn", force=True)
    main()
