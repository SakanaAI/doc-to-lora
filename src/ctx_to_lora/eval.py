import yaml
import sys
import gc
import json
import os
import random
import shutil
from argparse import Namespace
from dataclasses import fields
from functools import partial
from collections import defaultdict
from typing import Callable

import numpy as np
import torch
from datasets import disable_caching
from peft import PeftModel
from rouge_score import rouge_scorer
from transformers import (
    PreTrainedModel,
    GenerationConfig,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    EvalPrediction,
    set_seed,
)
from transformers.utils import is_liger_kernel_available


from ctx_to_lora.data_utils import get_tokenized_dataset
from ctx_to_lora.modeling_utils import ModulatedPretrainedModel
from ctx_to_lora.model_loading import get_tokenizer, get_model


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


def compute_rouge(pred_texts, label_texts):
    out = defaultdict(list)
    scorer = rouge_scorer.RougeScorer(["rouge1", "rougeL"], use_stemmer=False)
    for pred_text, label_text in zip(pred_texts, label_texts):
        scores = scorer.score(pred_text, label_text)
        for k, v in scores.items():
            out[f"{k}.f1"].append(v.fmeasure)
    for k in out:
        out[k] = np.mean(out[k])
    return out


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


@torch.inference_mode()
def compute_metrics(
    eval_pred: EvalPrediction,
    compute_result: bool,
    evaluator: Evaluator,
):
    logits, labels = eval_pred.predictions, eval_pred.label_ids
    shift_logits = logits[..., :-1, :]
    shift_labels = labels[..., 1:]
    valid_masks = torch.where(shift_labels != -100, 1, 0)
    evaluator.update(shift_logits, shift_labels, valid_masks)
    if compute_result:
        return evaluator.compute()


def save_generated_text(samples, output_dir, split):
    os.makedirs(output_dir, exist_ok=True)
    with open(f"{output_dir}/{split}_generated_text.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def decode_test_result(test_dataset, test_result, tokenizer, ctx_tokenizer):
    out = []
    for sample, pred_toks in zip(test_dataset, test_result.predictions):
        # sample and labels are not padded
        # pred_toks are padded though
        d = dict()
        if "labels" in sample:
            start_idx = np.argmax(sample["labels"] != -100)
            label_toks = sample["labels"][start_idx:]
            # labels are padded with -100, so we need to
            # replace them with the pad token id
            label_toks = np.where(label_toks == -100, tokenizer.pad_token_id, label_toks)
            label_text = tokenizer.decode(label_toks, skip_special_tokens=True)
            d["label"] = label_text

        # remove the label part
        input_toks = sample["input_ids"][:start_idx]

        gen_toks = pred_toks[np.argmax(pred_toks != tokenizer.pad_token_id) :]
        gen_toks = gen_toks[start_idx:]
        gen_toks = np.where(gen_toks == -100, tokenizer.pad_token_id, gen_toks)

        d["input"] = tokenizer.decode(input_toks, skip_special_tokens=True)
        d["generated"] = tokenizer.decode(gen_toks, skip_special_tokens=True)
        if "ctx_ids" in sample:
            d["context"] = ctx_tokenizer.decode(
                sample["ctx_ids"], skip_special_tokens=True
            )
        out.append(d)

    return out


@torch.inference_mode()
def eval_generation(eval_trainer, tokenizer, ctx_tokenizer, datasets, split, gen_kwargs):
    if not isinstance(datasets, dict):
        datasets = {"": datasets}
    out = {}
    for ds_name, ds in datasets.items():
        split_name = f"{split}_{ds_name}" if ds_name else split
        eval_result = eval_trainer.predict(
            ds,
            metric_key_prefix=split_name,
            **gen_kwargs,
        )
        decoded_txts = decode_test_result(ds, eval_result, tokenizer, ctx_tokenizer)
        rouge_metrics = compute_rouge(
            [txt["generated"] for txt in decoded_txts],
            [txt["label"] for txt in decoded_txts],
        )
        for k, v in rouge_metrics.items():
            eval_result.metrics[f"{split_name}_{k}"] = v
        save_generated_text(
            decoded_txts,
            split=split_name,
            output_dir=eval_trainer.args.output_dir,
        )
        out[split_name] = eval_result.metrics
        eval_trainer.log_metrics(split_name, eval_result.metrics)
        eval_trainer.save_metrics(split_name, eval_result.metrics)
        clear_gpu()
    return out


@torch.no_grad()
def eval_teacher_forcing(eval_trainer, datasets, split):
    if not isinstance(datasets, dict):
        datasets = {"": datasets}
    out = {}
    for ds_name, ds in datasets.items():
        split_name = f"{split}_{ds_name}" if ds_name else split
        metrics = eval_trainer.evaluate(ds, metric_key_prefix=split_name)
        out[split_name] = metrics
        eval_trainer.log_metrics(split_name, metrics)
        eval_trainer.save_metrics(split_name, metrics)
        clear_gpu()
    return out


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

    chat_ids = None
    if "chat_ids" in inp_list[0]:
        chat_ids = [x.pop("chat_ids") for x in inp_list]
        chat_ids = torch.nn.utils.rnn.pad_sequence(
            chat_ids,
            batch_first=True,
            padding_value=0,
        )
        chat_attn_mask = [x.pop("chat_attn_mask") for x in inp_list]
        chat_attn_mask = torch.nn.utils.rnn.pad_sequence(
            chat_attn_mask,
            batch_first=True,
            padding_value=0,
        )
        chat_labels = [x.pop("chat_labels") for x in inp_list]
        chat_labels = torch.nn.utils.rnn.pad_sequence(
            chat_labels,
            batch_first=True,
            padding_value=-100,
        )
        chat_labels = torch.where(chat_attn_mask == 0, -100, chat_labels)

    labels = [x.pop("labels") for x in inp_list]
    padded_seq = tokenizer.pad(inp_list, **padding_kwargs)

    # hacky explicit padding since the labels are not padded by default
    labels = tokenizer.pad({"input_ids": labels}, **padding_kwargs)["input_ids"]
    labels = torch.where(padded_seq["attention_mask"] == 0, -100, labels)
    out = {**padded_seq, "labels": labels}
    if ctx_ids is not None:
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask
    if chat_ids is not None:
        out["chat_ids"] = chat_ids
        out["chat_attn_mask"] = chat_attn_mask
        out["chat_labels"] = chat_labels
    return out


def generation_collator(inp_list, tokenizer):
    padding_kwargs = dict(padding=True, padding_side="left", return_tensors="pt")
    input_ids = [x.pop("input_ids") for x in inp_list]
    attn_mask = [x.pop("attention_mask") for x in inp_list]
    labels = [x.pop("labels") for x in inp_list]
    for i, label in enumerate(labels):
        # remove the response tokens
        idx = np.argmax(label != -100)
        input_ids[i] = input_ids[i][:idx]
        attn_mask[i] = attn_mask[i][:idx]
    out = tokenizer.pad(
        {"input_ids": input_ids, "attention_mask": attn_mask}, **padding_kwargs
    )

    # we don't include the labels in the output
    # since we don't want to compute the loss on the labels
    # during generation
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
        out["ctx_ids"] = ctx_ids
        out["ctx_attn_mask"] = ctx_attn_mask
    return out


def evaluate(
    checkpoint_path, model_name_or_path, eval_batch_size, args, split, generative
):
    assert split in ["validation", "test"]
    ctx_name = None
    if model_name_or_path is None:
        state_dict = torch.load(checkpoint_path, weights_only=False)
        ctx_name = state_dict["ctx_encoder_args"].ctx_encoder_model_name_or_path
        model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            use_flash_attn=True,
        )
        if generative:
            model = model.to(torch.bfloat16)
    else:
        model = get_model(model_name_or_path, train=False, requires_grad=False)
    # NOTE: there is still some randomness in the eval result
    # despite all the deterministic settings
    if is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        if isinstance(model, ModulatedPretrainedModel):
            print("Applying liger-kernel to ModulatedPretrainedModel")
            if isinstance(model.base_model, PeftModel):
                _apply_liger_kernel_to_instance(model=model.base_model.base_model.model)
            else:
                _apply_liger_kernel_to_instance(model=model.base_model.model)
            if ctx_name is not None:
                print("Applying liger-kernel to ctx_encoder_model")
                _apply_liger_kernel_to_instance(model=model.ctx_encoder.base_model)
        elif isinstance(model, PeftModel):
            print("Applying liger-kernel to PeftModel")
            _apply_liger_kernel_to_instance(model=model.base_model.model)
        elif isinstance(model, PreTrainedModel):
            print("Applying liger-kernel to PretrainedModel")
            _apply_liger_kernel_to_instance(model=model.model)

    tokenizer = get_tokenizer(args.model_name_or_path, train=False)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token_id = tokenizer.eos_token_id
    base_model = (
        model.base_model if isinstance(model, ModulatedPretrainedModel) else model
    )
    base_model.config.pad_token_id = tokenizer.pad_token_id
    base_model.generation_config.pad_token_id = tokenizer.pad_token_id

    ctx_tokenizer = tokenizer
    if ctx_name:
        ctx_tokenizer = get_tokenizer(ctx_name, train=False)
        if ctx_tokenizer.pad_token_id is None:
            ctx_tokenizer.pad_token_id = ctx_tokenizer.eos_token_id

    tokenizer_kwargs = {"max_length": args.max_base_len}
    ctx_tokenizer_kwargs = {"max_length": args.max_ctx_len}
    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)

    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        ctx_tokenizer=ctx_tokenizer,
        ctx_tokenizer_kwargs=ctx_tokenizer_kwargs,
        add_ctx_to_chat=add_ctx_to_chat,
        add_repeat_prompt=False,
        add_negative_prompt=False,
        use_kl_loss=False,
        set_format="pt",
    )

    datasets = dict()
    ds_names = args.val_ds_names if split == "validation" else args.test_ds_names
    for ds_name in ds_names:
        datasets[ds_name] = _get_tokenized_dataset(ds_name, split)
    print(datasets)

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=256,  # max_new_tokens=args.max_new_tokens
    )

    eval_trainer_args = {}

    # Copy only necessary attributes from training_args to eval_trainer_args
    seq2seq_training_args_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    for attr, value in dict(**vars(args)).items():
        if attr in seq2seq_training_args_fields and not attr.startswith("_"):
            eval_trainer_args[attr] = value

    eval_trainer_args["eval_strategy"] = "no"
    eval_trainer_args["save_strategy"] = "no"
    eval_trainer_args["overwrite_output_dir"] = True
    eval_trainer_args["batch_eval_metrics"] = True
    eval_trainer_args["per_device_eval_batch_size"] = eval_batch_size

    eval_trainer_args = Seq2SeqTrainingArguments(
        **eval_trainer_args,
        predict_with_generate=generative,
        generation_config=GenerationConfig(**gen_kwargs),
    )

    # Seq2SeqTrainer is actually just the same as Trainer
    # (although it uses a different data collator, i.e., explicit prompt/answer separation)
    # it just allows `predict_with_generate`
    # allowing us to compute metrics on the generated outputs
    # no clue why they call this seq2seq...

    print("=" * 80 + "\n" + "Evaluating model..." + "\n" + "=" * 80)
    print(f"checkpoint_path: {checkpoint_path}")

    model.eval()

    collator = generation_collator if generative else train_collator

    trainer_kwargs = {
        "model": model,
        "args": eval_trainer_args,
        "data_collator": partial(collator, tokenizer=tokenizer),
    }
    out = {}
    if not generative:

        trainer_kwargs["compute_metrics"] = partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_perplexity]
            ),
        )
        # this is insane
        # but i don't know why calling trainer.evaluate() on different datasets
        # always gives the same numbers across datasets...
        # spents a few hours on this but couldn't find the reason
        for ds_name, ds in datasets.items():
            eval_trainer = Trainer(**trainer_kwargs)
            metrics = eval_teacher_forcing(eval_trainer, {ds_name: ds}, split)
            out.update(metrics)
    else:
        eval_trainer = Seq2SeqTrainer(**trainer_kwargs)
        metrics = eval_generation(
            eval_trainer, tokenizer, ctx_tokenizer, datasets, split, gen_kwargs
        )
        out.update(metrics)

    clear_gpu()
    return out


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Evaluate a checkpoint")
    parser.add_argument(
        "--model_name_or_path",
        type=str,
        default=None,
        help="Evaluate a base model from HuggingFace Hub, without loading checkpoint",
    )
    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default=None,
        help="Path to the checkpoint to evaluate",
    )
    parser.add_argument(
        "--split",
        type=str,
        choices=["validation", "test"],
        default="validation",
        help="Which split to evaluate on",
    )
    parser.add_argument(
        "--datasets",
        type=str,
        nargs="+",
        help=(
            "Specific datasets to evaluate on."
            "If not provided, uses default from args.yaml"
        ),
    )
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=8,
        help="Eval batch size for teacher forcing",
    )
    parser.add_argument(
        "--eval_batch_size_gen",
        type=int,
        default=32,
        help="Eval batch size for generation",
    )

    cli_args = parser.parse_args()

    assert bool(cli_args.model_name_or_path) ^ bool(
        cli_args.checkpoint_path
    ), "Either --model_name_or_path or --checkpoint_path must be provided"

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    disable_caching()
    set_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if cli_args.checkpoint_path:
        checkpoint_dir = "/".join(cli_args.checkpoint_path.split("/")[:-1])
        run_dir = "/".join(cli_args.checkpoint_path.split("/")[:-2])
        cur_it = int(cli_args.checkpoint_path.split("checkpoint-")[1].split("/")[0])
        args = Namespace(**yaml.unsafe_load(open(f"{run_dir}/args.yaml", "r")))
        print(f"checkpoint_path: {cli_args.checkpoint_path}")
        print(f"run_dir: {run_dir}")

        args.output_dir = f"{run_dir}/eval-results-{cur_it}"
        args.logging_dir = f"{run_dir}/eval-results-{cur_it}"
        args.run_name = run_dir.split("/")[-1]
    else:
        args = Namespace(
            model_name_or_path=cli_args.model_name_or_path,
            output_dir=f"eval_results/{cli_args.model_name_or_path}",
            logging_dir=f"eval_results/{cli_args.model_name_or_path}",
            run_name=f"eval_results/{cli_args.model_name_or_path}",
            max_base_len=2**13,
            max_ctx_len=-1,  # not used
            val_ds_names=[],
            test_ds_names=[],
        )

    # Override dataset names if provided via CLI
    if cli_args.datasets:
        if cli_args.split == "validation":
            args.val_ds_names = cli_args.datasets
        else:
            args.test_ds_names = cli_args.datasets

    # evaluate(
    #     cli_args.checkpoint_path,
    #     cli_args.model_name_or_path,
    #     cli_args.eval_batch_size,
    #     args,
    #     split=cli_args.split,
    #     generative=False,
    # )
    evaluate(
        cli_args.checkpoint_path,
        cli_args.model_name_or_path,
        cli_args.eval_batch_size_gen,
        args,
        split=cli_args.split,
        generative=True,
    )
