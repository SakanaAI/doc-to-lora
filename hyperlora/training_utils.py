import gc
import json
import logging
from collections import defaultdict
from dataclasses import fields
from enum import Enum

import numpy as np
from rouge_score import rouge_scorer
import torch
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
)
from transformers.trainer_utils import get_last_checkpoint

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])

logger = logging.getLogger()


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


def save_generated_text(samples, output_dir, split):
    with open(f"{output_dir}/{split}_generated_text.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def decode_test_result(test_dataset, test_result, tokenizer):
    out = []
    for sample, pred_toks in zip(test_dataset, test_result.predictions):
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
        gen_toks = pred_toks[len(input_toks) :]
        gen_toks = np.where(gen_toks == -100, tokenizer.pad_token_id, gen_toks)

        d["input"] = tokenizer.decode(input_toks, skip_special_tokens=True)
        d["generated"] = tokenizer.decode(gen_toks, skip_special_tokens=True)
        if "ctx_ids" in sample:
            d["context"] = tokenizer.decode(sample["ctx_ids"], skip_special_tokens=True)
        out.append(d)

    return out


def eval_generation(eval_trainer, tokenizer, dataset, split, gen_kwargs):
    if not isinstance(dataset, dict):
        dataset = {"": dataset}

    for ds_name, ds in dataset.items():
        split_name = f"{split}_{ds_name}" if ds_name else split
        eval_result = eval_trainer.predict(
            ds,
            metric_key_prefix=split_name,
            **gen_kwargs,
        )
        decoded_txts = decode_test_result(ds, eval_result, tokenizer)
        rouge_metrics = compute_rouge(
            [txt["generated"] for txt in decoded_txts],
            [txt["label"] for txt in decoded_txts],
        )
        for k, v in rouge_metrics.items():
            eval_result.metrics[f"{split}_{k}"] = v

        save_generated_text(
            decoded_txts,
            split=split_name,
            output_dir=eval_trainer.args.output_dir,
        )
        eval_trainer.log_metrics(split_name, eval_result.metrics)
        eval_trainer.save_metrics(split_name, eval_result.metrics)
        clear_gpu()


# def per_sample_loss_avg_fn(outputs, labels, num_items_in_batch):
#     ...


def train_model(
    model,
    tokenizer,
    training_args,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    train_collator=None,
    generation_collator=None,
    compute_metrics=None,
    preprocess_logits_for_metrics=None,
    per_sample_loss_avg=False,
    max_new_tokens=2**13,
    gen_per_device_eval_batch_size=1,
):

    # last_checkpoint = None
    # if (
    #     os.path.isdir(training_args.output_dir)
    #     and not training_args.overwrite_output_dir
    # ):
    #     last_checkpoint = get_last_checkpoint(training_args.output_dir)
    #     if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
    #         raise ValueError(
    #             f"Output directory ({training_args.output_dir})"
    #             " already exists and is not empty. "
    #             "Use --overwrite_output_dir to overcome."
    #         )
    #     elif (
    #         last_checkpoint is not None and training_args.resume_from_checkpoint is None
    #     ):
    #         print(
    #             f"Checkpoint detected, resuming training at {last_checkpoint}. "
    #             "To avoid this behavior, change "
    #             "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
    #         )

    # if (
    #     max(
    #         training_args.per_device_train_batch_size,
    #         training_args.per_device_eval_batch_size,
    #     )
    #     == 1
    # ):
    #     data_collator = None

    # # print training_args at local_rank 0
    # local_rank = int(os.getenv("LOCAL_RANK", "0"))
    # if local_rank == 0:
    #     print(training_args)

    compute_loss_func = per_sample_loss_avg_fn if per_sample_loss_avg else None

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    checkpoint = None

    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint

    # print(f"Loaded from the checkpoint: {checkpoint}")

    # Trainer loads the best model after training
    # is done when load_best_model_at_end=True (our default)
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate(dict(**val_dataset, test=test_dataset))
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model()

    clear_gpu()

    ############## Evaluation
    # max_input_len=2**13 # for input truncation

    gen_kwargs = dict(do_sample=False, max_new_tokens=max_new_tokens)
    # pad_token_id=tokenizer.pad_token_id,
    # eos_token_id=?

    eval_trainer_args = {}

    # Copy only necessary attributes from training_args to eval_trainer_args
    seq2seq_training_args_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    for attr, value in training_args.to_dict().items():
        if attr in seq2seq_training_args_fields:
            eval_trainer_args[attr] = value

    eval_trainer_args["eval_strategy"] = "no"
    eval_trainer_args["save_strategy"] = "no"
    eval_trainer_args["overwrite_output_dir"] = True
    eval_trainer_args["per_device_eval_batch_size"] = gen_per_device_eval_batch_size

    # NOTE: could also set kv_cache implementation here
    eval_trainer_args = Seq2SeqTrainingArguments(
        **eval_trainer_args,
        predict_with_generate=True,
        generation_config=GenerationConfig(**gen_kwargs),
    )

    # Seq2SeqTrainer is actually just the same as Trainer
    # (although it uses a different data collator, i.e., explicit prompt/answer separation)
    # it just allows `predict_with_generate`
    # allowing us to compute metrics on the generated outputs
    # no clue why they call this seq2seq...

    logger.info("=" * 80 + "\n" + "Evaluating model..." + "\n" + "=" * 80)

    model.eval()

    eval_trainer = Seq2SeqTrainer(
        model=model,
        args=eval_trainer_args,
        # TODO: use a different collator for test, e.g., more max_len truncation
        # w/ left padding?
        # removing label part from input_ids
        data_collator=generation_collator,
    )

    for split, ds in zip(["eval", "test"], [val_dataset, test_dataset]):
        if ds is None:
            continue
        eval_generation(eval_trainer, tokenizer, ds, split, gen_kwargs)
        clear_gpu()

    # if test_dataset is not None:
    #     eval_generation(eval_trainer, tokenizer, test_dataset, "test", gen_kwargs)
