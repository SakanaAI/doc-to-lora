import json
import logging
import os
import re
import string
import sys
from argparse import Namespace
from collections import Counter, defaultdict
from dataclasses import fields
from functools import partial

import numpy as np
import pandas as pd
import torch
import yaml
from datasets import disable_caching
from peft import PeftModel
from transformers import (
    GenerationConfig,
    PreTrainedModel,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
    set_seed,
)
from transformers.utils import is_liger_kernel_available

from ctx_to_lora.data.collator import eval_collator, generation_collator
from ctx_to_lora.data.definitions import (
    CLOSED_QA_DATASETS,
    LONGBENCH_E_TASKS,
    LONGBENCH_TASKS,
)
from ctx_to_lora.data.processing import get_tokenized_dataset
from ctx_to_lora.metrics import (
    LENGTH_BINS,
    Evaluator,
    compute_metrics,
    compute_per_token_acc,
    compute_perplexity,
    compute_prefix_matching,
    compute_rouge,
)
from ctx_to_lora.model_loading import get_model, get_tokenizer
from ctx_to_lora.modeling import hypernet
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.utils import clear_gpu, setup_logging

logger = logging.getLogger()


# bandaid for loading old models (before restructure)
sys.modules["ctx_to_lora.modeling_utils"] = hypernet


# ============================================================================
# Metrics and Evaluation Utilities
# ============================================================================


def normalize_answer(s: str) -> str:
    """Lower text and remove punctuation, articles and extra whitespace."""

    def remove_articles(text: str) -> str:
        return re.sub(r"\b(a|an|the)\b", " ", text)

    def white_space_fix(text: str) -> str:
        return " ".join(text.split())

    def remove_punc(text: str) -> str:
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)

    def lower(text: str) -> str:
        return text.lower()

    return white_space_fix(remove_articles(remove_punc(lower(s))))


def f1_score(prediction: str, ground_truth: str) -> float:
    """Compute F1 score between prediction and ground truth strings."""
    common = Counter(prediction) & Counter(ground_truth)
    num_same = sum(common.values())
    if num_same == 0:
        return 0
    precision = 1.0 * num_same / len(prediction)
    recall = 1.0 * num_same / len(ground_truth)
    f1 = (2 * precision * recall) / (precision + recall)
    return f1


def compute_qa_f1_score(
    pred_texts: list[str], label_texts: list[str]
) -> dict[str, float]:
    """
    Word-level F1 score for evaluating question answering systems.
    Order of the words does not matter.
    """
    res = []
    for prediction, label in zip(pred_texts, label_texts):
        normalized_prediction = normalize_answer(prediction)
        normalized_label = normalize_answer(label)

        prediction_words = normalized_prediction.split()
        label_words = normalized_label.split()
        res.append(f1_score(prediction_words, label_words))
    return dict(qa_f1=np.mean(res))


def add_longbench_tasks(ds_names: list[str]) -> None:
    """Add longbench tasks to dataset names list."""
    if "longbench" in ds_names:
        ds_names.remove("longbench")
        ds_names += LONGBENCH_TASKS
    if "longbench_e" in ds_names:
        ds_names.remove("longbench_e")
        ds_names += LONGBENCH_E_TASKS


def save_generated_text(samples: list[dict], output_dir: str, split: str) -> None:
    """Save generated text samples to JSONL file."""
    os.makedirs(output_dir, exist_ok=True)
    # Create any necessary subdirectories if split contains path separators
    if "/" in split:
        split_dir = os.path.join(output_dir, os.path.dirname(split))
        os.makedirs(split_dir, exist_ok=True)

    with open(f"{output_dir}/{split}_generated_text.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


# ============================================================================
# CSV Export Utilities
# ============================================================================


def _extract_model_info(eval_trainer) -> tuple[str, bool]:
    """Extract model name and type information from the trainer."""
    model_name = "unknown_model"
    is_hypernet = False

    if hasattr(eval_trainer.model, "base_model"):
        if hasattr(eval_trainer.model.base_model, "config"):
            model_name = getattr(
                eval_trainer.model.base_model.config,
                "name_or_path",
                getattr(
                    eval_trainer.model.base_model.config, "_name_or_path", "unknown"
                ),
            )
        is_hypernet = hasattr(eval_trainer.model, "ctx_encoder")
    elif hasattr(eval_trainer.model, "config"):
        model_name = getattr(
            eval_trainer.model.config,
            "name_or_path",
            getattr(eval_trainer.model.config, "_name_or_path", "unknown"),
        )

    # Clean up model name for display
    if "/" in model_name:
        model_name = model_name.split("/")[-1]

    if getattr(eval_trainer.args, "run_name", None):
        model_name += f"_{eval_trainer.args.run_name}"

    return model_name, is_hypernet


def _parse_metrics_for_csv(
    metrics_dict: dict[str, dict[str, any]],
) -> tuple[set, set, set]:
    """Parse metrics dictionary to extract unique metrics, length groups, and splits."""
    all_metrics = set()
    all_length_groups = set()
    all_splits = set()

    for split_name, metrics in metrics_dict.items():
        all_splits.add(split_name)

        for metric_key in metrics.keys():
            if not metric_key.startswith(split_name):
                continue

            # Skip timing and performance metrics that aren't evaluation results
            if any(
                skip_term in metric_key
                for skip_term in [
                    "model_preparation_time",
                    "steps_per_second",
                    "samples_per_second",
                    "runtime",
                ]
            ):
                continue

            # Remove the split prefix to get the actual metric name
            metric_name = metric_key[len(split_name) + 1 :]

            # Check if this is a length-specific metric
            if "_len_" in metric_name:
                base_metric, length_part = metric_name.split("_len_", 1)
                all_metrics.add(base_metric)
                all_length_groups.add(length_part)
            else:
                all_metrics.add(metric_name)

    # Add overall metric (no length grouping)
    all_length_groups.add("overall")
    return all_metrics, all_length_groups, all_splits


def _sort_length_groups(length_groups: set[str]) -> list[str]:
    """Sort length groups with custom ordering for proper numerical ranges."""

    def sort_key(length_group: str) -> tuple:
        if length_group == "overall":
            return (1, 0, 0)  # Put "overall" after numerical ranges
        try:
            # Parse "low-high" format
            low, high = map(float, length_group.split("-"))
            return (0, low, high)
        except (ValueError, IndexError):
            return (2, 0, 0)  # Put any malformed strings last

    return sorted(list(length_groups), key=sort_key)


def create_metrics_csv(
    metrics_dict: dict[str, dict[str, any]],
    output_dir: str,
    model_name: str,
    is_hypernet_model: bool = False,
    remove_context: bool = False,
    csv_suffix: str = "",
) -> None:
    """
    Create a CSV file with columns: model_name, group_len, tasks, num_samples, and all available metrics.
    One row per model-length-task combination.

    Args:
        metrics_dict: Dictionary containing evaluation metrics for each dataset/split
        output_dir: Directory to save the CSV file
        model_name: Name of the model being evaluated
        is_hypernet_model: Whether this is a hypernet/modulated model or base model
        remove_context: Whether context was removed during evaluation
        csv_suffix: Additional suffix for the CSV filename
    """
    os.makedirs(output_dir, exist_ok=True)

    # Parse metrics to extract components
    all_metrics, all_length_groups, all_splits = _parse_metrics_for_csv(metrics_dict)

    # Sort for consistent ordering
    all_length_groups = _sort_length_groups(all_length_groups)
    all_splits = sorted(all_splits)

    # Create rows for each model-length-task combination
    rows = []

    for task in all_splits:
        metrics = metrics_dict[task]

        for length_group in all_length_groups:
            # Initialize row with basic info
            row = {
                "model_name": model_name,
                "group_len": length_group,
                "tasks": task,
                "num_samples": 0,
            }

            # Look for all metrics for this task and length group
            if length_group == "overall":
                # Look for overall metrics (no length suffix)
                for metric_key in metrics:
                    if not metric_key.startswith(f"{task}_"):
                        continue

                    # Skip length-specific metrics
                    if "_len_" in metric_key:
                        continue

                    # Extract metric name after task prefix
                    metric_name = metric_key[len(task) + 1 :]
                    if metric_name in [
                        "samples_per_second",
                        "steps_per_second",
                        "model_preparation_time",
                        "runtime",
                    ]:
                        # Skip timing and performance metrics
                        continue

                    # Handle num_samples specially
                    if metric_name.startswith("num_samples_"):
                        row["num_samples"] = metrics[metric_key]
                    else:
                        # Add all other metrics as columns
                        row[metric_name] = metrics[metric_key]
            else:
                # Look for length-specific metrics
                for metric_key in metrics:
                    if not metric_key.startswith(f"{task}_"):
                        continue

                    # Only process metrics for this specific length group
                    if f"_len_{length_group}" not in metric_key:
                        continue

                    # Extract metric name (remove task prefix and length suffix)
                    metric_part = metric_key[len(task) + 1 :]
                    metric_name = metric_part.replace(f"_len_{length_group}", "")
                    if metric_name in [
                        "samples_per_second",
                        "steps_per_second",
                        "model_preparation_time",
                        "runtime",
                    ]:
                        # Skip timing and performance metrics
                        continue

                    # Handle num_samples specially
                    if metric_name.startswith("num_samples_"):
                        row["num_samples"] = metrics[metric_key]
                    else:
                        # Add all other metrics as columns
                        row[metric_name] = metrics[metric_key]

            # Fill missing metrics with N/A for consistent columns
            for metric in all_metrics:
                if metric not in row and not metric.startswith("num_samples"):
                    row[metric] = "N/A"

            rows.append(row)

    # Create DataFrame
    if rows:
        df = pd.DataFrame(rows)

        # Sort by tasks, then by length group
        df = df.sort_values(["tasks"]).reset_index(drop=True)

        # Construct filename
        csv_filename = "evaluation_results"
        if csv_suffix:
            csv_filename += f"_{csv_suffix}"
        if remove_context:
            csv_filename += "_no_context"
        csv_filename += ".csv"

        csv_path = os.path.join(output_dir, csv_filename)
        df.to_csv(csv_path, index=False)
        print(f"Evaluation results saved to: {csv_path}")
    else:
        print("No evaluation data found to save to CSV")


# ============================================================================
# Evaluation Functions
# ============================================================================


def decode_test_result(
    test_dataset, test_result, tokenizer, ctx_tokenizer
) -> list[dict]:
    """Decode test results into human-readable format."""
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
            label_toks = np.where(
                label_toks == -100, tokenizer.pad_token_id, label_toks
            )
            label_text = tokenizer.decode(label_toks, skip_special_tokens=True)
            d["label"] = label_text.strip()

        # remove the label part
        input_toks = sample["input_ids"][:start_idx]

        gen_toks = pred_toks[np.argmax(pred_toks != tokenizer.pad_token_id) :]
        gen_toks = gen_toks[start_idx:]
        gen_toks = np.where(gen_toks == -100, tokenizer.pad_token_id, gen_toks)

        d["input"] = tokenizer.decode(input_toks, skip_special_tokens=True)
        d["generated"] = tokenizer.decode(gen_toks, skip_special_tokens=True).strip()
        if "ctx_ids" in sample:
            d["context"] = ctx_tokenizer.decode(
                sample["ctx_ids"], skip_special_tokens=True
            )
        for k in sample:
            if k.endswith("_len"):
                d[k] = sample[k].item()
        out.append(d)

    # sort samples by length if possible
    len_key = "ctx_ids_len" if "ctx_ids_len" in sample else "input_ids_len"
    sorted(out, key=lambda x: x[len_key])

    return out


@torch.inference_mode()
def eval_generation(
    eval_trainer, tokenizer, ctx_tokenizer, datasets, split, remove_context, gen_kwargs
) -> dict[str, dict]:
    """Evaluate model using generation and save metrics to CSV."""
    if not isinstance(datasets, dict):
        datasets = {"": datasets}

    out = {}
    for ds_name, ds in datasets.items():
        print(f"Evaluating: {ds_name}")
        split_name = f"{split}_{ds_name}" if ds_name else split
        if remove_context:
            split_name += "_no_context"

        eval_result = eval_trainer.predict(
            ds,
            metric_key_prefix=split_name,
            **gen_kwargs,
        )
        decoded_txts = decode_test_result(ds, eval_result, tokenizer, ctx_tokenizer)
        pred_texts = [txt["generated"] for txt in decoded_txts]
        label_texts = [txt["label"] for txt in decoded_txts]
        n = len(pred_texts)

        if ds_name in CLOSED_QA_DATASETS:
            print("Computing QA F1 Score")
            qa_f1_metric = compute_qa_f1_score(
                [txt["generated"] for txt in decoded_txts],
                [txt["label"] for txt in decoded_txts],
            )
            for k, v in qa_f1_metric.items():
                eval_result.metrics[f"{split_name}_{k}"] = v
                eval_result.metrics[f"{split_name}_num_samples_{k}"] = n
        else:
            rouge_metrics = compute_rouge(pred_texts, label_texts)
            for k, v in rouge_metrics.items():
                eval_result.metrics[f"{split_name}_{k}"] = v
                eval_result.metrics[f"{split_name}_num_samples_{k}"] = n

        # Ensure all keys for length metrics are present, even if a bin is empty
        for low, high in LENGTH_BINS:
            if ds_name in CLOSED_QA_DATASETS:
                eval_result.metrics[f"{split_name}_qa_f1_len_{low}-{high}"] = "None"
            else:
                eval_result.metrics[f"{split_name}_rougeL.f1_len_{low}-{high}"] = "None"

        grouped_texts = defaultdict(lambda: {"generated": [], "label": [], "count": 0})
        for txt in decoded_txts:
            len_key = "ctx_ids_len" if "ctx_ids_len" in txt else "input_ids_len"
            input_len = txt[len_key]
            for low, high in LENGTH_BINS:
                if low <= input_len <= high:
                    group_key = f"{low}-{high}"
                    grouped_texts[group_key]["generated"].append(txt["generated"])
                    grouped_texts[group_key]["label"].append(txt["label"])
                    grouped_texts[group_key]["count"] += 1
                    break

        for group_key, data in grouped_texts.items():
            if data["count"] > 0:
                if ds_name in CLOSED_QA_DATASETS:
                    group_qa_f1_metric = compute_qa_f1_score(
                        data["generated"], data["label"]
                    )
                    for k, v in group_qa_f1_metric.items():
                        eval_result.metrics[f"{split_name}_{k}_len_{group_key}"] = v
                        eval_result.metrics[
                            f"{split_name}_num_samples_{k}_len_{group_key}"
                        ] = data["count"]
                else:
                    group_rouge_metrics = compute_rouge(
                        data["generated"], data["label"]
                    )
                    for k, v in group_rouge_metrics.items():
                        eval_result.metrics[f"{split_name}_{k}_len_{group_key}"] = v
                        # also add count data
                        eval_result.metrics[
                            f"{split_name}_num_samples_{k}_len_{group_key}"
                        ] = data["count"]

        save_generated_text(
            decoded_txts,
            split=split_name,
            output_dir=eval_trainer.args.output_dir,
        )
        out[split_name] = eval_result.metrics
        eval_trainer.log_metrics(split_name, eval_result.metrics)
        eval_trainer.save_metrics(split_name, eval_result.metrics)
        clear_gpu()

    # Create CSV summary of all evaluation results
    if out:
        model_name, is_hypernet = _extract_model_info(eval_trainer)
        create_metrics_csv(
            out,
            output_dir=eval_trainer.args.output_dir,
            model_name=model_name,
            is_hypernet_model=is_hypernet,
            remove_context=remove_context,
            csv_suffix="generation",
        )

    return out


@torch.no_grad()
def eval_teacher_forcing(
    eval_trainer, datasets, split, remove_context
) -> dict[str, dict]:
    """Evaluate using teacher forcing and save metrics to CSV."""
    if not isinstance(datasets, dict):
        datasets = {"": datasets}

    out = {}
    for ds_name, ds in datasets.items():
        split_name = f"{split}_{ds_name}" if ds_name else split
        if remove_context:
            split_name += "_no_context"

        metrics = eval_trainer.evaluate(ds, metric_key_prefix=split_name)
        out[split_name] = metrics
        eval_trainer.log_metrics(split_name, metrics)
        eval_trainer.save_metrics(split_name, metrics)
        clear_gpu()

    # Create CSV for teacher forcing metrics
    if out:
        model_name, is_hypernet = _extract_model_info(eval_trainer)
        create_metrics_csv(
            out,
            output_dir=eval_trainer.args.output_dir,
            model_name=model_name,
            is_hypernet_model=is_hypernet,
            remove_context=remove_context,
            csv_suffix="teacher_forcing",
        )

    return out


def evaluate(
    checkpoint_path: str,
    model_name_or_path: str,
    eval_batch_size: int,
    args: Namespace,
    split: str,
    generative: bool,
) -> dict[str, dict]:
    """Main evaluation function."""
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
    else:
        model_kwargs = dict(attn_implementation="flash_attention_2")
        model = get_model(
            model_name_or_path,
            train=False,
            requires_grad=False,
            model_kwargs=model_kwargs,
        )

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

    tokenizer_kwargs = {}
    ctx_tokenizer_kwargs = {}
    add_ctx_to_chat = (
        not isinstance(model, ModulatedPretrainedModel) and not args.remove_context
    )
    ctx_model_max_len = (
        model.ctx_encoder.config.max_position_embeddings
        if isinstance(model, ModulatedPretrainedModel)
        else None
    )

    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        ctx_model_max_len=ctx_model_max_len,
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
    add_longbench_tasks(ds_names)
    for ds_name in ds_names:
        datasets[ds_name] = _get_tokenized_dataset(ds_name, split)
    print(datasets)

    # truncating num val samples
    max_eval_samples_per_ds = getattr(args, "max_val_samples_per_ds", 0)
    if split == "validation" and max_eval_samples_per_ds > 0:
        print(f"Truncating all validation ds to {max_eval_samples_per_ds} samples")
        for ds_name, ds in datasets.items():
            val_indices = np.random.permutation(len(ds))[:max_eval_samples_per_ds]
            datasets[ds_name] = ds.select(val_indices)
    print(datasets)

    gen_kwargs = dict(
        do_sample=False,
        max_new_tokens=256,
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
    eval_trainer_args["include_for_metrics"] = ["inputs"]

    eval_trainer_args = Seq2SeqTrainingArguments(
        **eval_trainer_args,
        predict_with_generate=generative,
        generation_config=GenerationConfig(**gen_kwargs),
    )

    print("=" * 80 + "\n" + "Evaluating model..." + "\n" + "=" * 80)
    print(f"checkpoint_path: {checkpoint_path}")

    model.eval()

    collator = generation_collator if generative else eval_collator

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
            metrics = eval_teacher_forcing(
                eval_trainer, {ds_name: ds}, split, args.remove_context
            )
            out.update(metrics)
    else:
        eval_trainer = Seq2SeqTrainer(**trainer_kwargs)
        metrics = eval_generation(
            eval_trainer,
            tokenizer,
            ctx_tokenizer,
            datasets,
            split,
            args.remove_context,
            gen_kwargs,
        )
        out.update(metrics)

    clear_gpu()
    return out


def run_eval(
    checkpoint_path: str = None,
    model_name_or_path: str = None,
    datasets: list[str] = None,
    split: str = "validation",
    eval_batch_size: int = 8,
    remove_context: bool = False,
    generative: bool = False,
) -> None:
    """Run evaluation with the specified parameters."""
    assert bool(model_name_or_path) ^ bool(checkpoint_path), (
        "Either --model_name_or_path or --checkpoint_path must be provided"
    )

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    disable_caching()
    set_seed(42)
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    if checkpoint_path:
        checkpoint_dir = "/".join(checkpoint_path.split("/")[:-1])
        run_dir = "/".join(checkpoint_path.split("/")[:-2])
        cur_it = int(checkpoint_path.split("checkpoint-")[1].split("/")[0])
        args = Namespace(**yaml.unsafe_load(open(f"{run_dir}/args.yaml")))
        print(f"checkpoint_path: {checkpoint_path}")
        print(f"run_dir: {run_dir}")

        args.output_dir = f"{run_dir}/eval-results-{cur_it}"
        args.logging_dir = f"{run_dir}/eval-results-{cur_it}"
        args.run_name = run_dir.split("/")[-1]
        # modulated model doesn't see ctx by default
        # but remove_context has to be false for correct file naming
        args.remove_context = False
    else:
        args = Namespace(
            model_name_or_path=model_name_or_path,
            output_dir=f"eval_results/{model_name_or_path}",
            logging_dir=f"eval_results/{model_name_or_path}",
            run_name=f"eval_results/{model_name_or_path}",
            val_ds_names=[],
            test_ds_names=[],
            remove_context=remove_context,
        )
    setup_logging(args.logging_dir)

    # Override dataset names if provided via CLI
    if datasets:
        if split == "validation":
            args.val_ds_names = datasets
        else:
            args.test_ds_names = datasets

    evaluate(
        checkpoint_path,
        model_name_or_path,
        eval_batch_size,
        args,
        split,
        generative=generative,
    )
