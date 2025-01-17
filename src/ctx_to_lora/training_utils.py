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


def train_model(
    model,
    # tokenizer,
    training_args,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    train_collator=None,
    # generation_collator=None,
    compute_metrics=None,
    # preprocess_logits_for_metrics=None,
    # max_new_tokens=2**13,
    # gen_per_device_eval_batch_size=1,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from the checkpoint: {checkpoint}")

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
        # preprocess_logits_for_metrics=preprocess_logits_for_metrics,
    )

    # Trainer loads the best model after training
    # is done when load_best_model_at_end=True
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_model()
    clear_gpu()

    # metrics = trainer.evaluate(dict(**val_dataset, test=test_dataset))
    # trainer.log_metrics("eval", metrics)
    # trainer.save_metrics("eval", metrics)
    # trainer.save_model()
    # clear_gpu()

    # ############## Evaluation
    # # TODO: eval does not work when using with deepspeed
    # # make a separate eval script

    # # max_input_len=2**13 # for input truncation

    # gen_kwargs = dict(do_sample=False, max_new_tokens=max_new_tokens)
    # # pad_token_id=tokenizer.pad_token_id,
    # # eos_token_id=?

    # eval_trainer_args = {}

    # # Copy only necessary attributes from training_args to eval_trainer_args
    # seq2seq_training_args_fields = {f.name for f in fields(Seq2SeqTrainingArguments)}
    # for attr, value in training_args.to_dict().items():
    #     if attr in seq2seq_training_args_fields:
    #         eval_trainer_args[attr] = value

    # eval_trainer_args["eval_strategy"] = "no"
    # eval_trainer_args["save_strategy"] = "no"
    # eval_trainer_args["overwrite_output_dir"] = True
    # eval_trainer_args["per_device_eval_batch_size"] = gen_per_device_eval_batch_size

    # # NOTE: could also set kv_cache implementation here
    # eval_trainer_args = Seq2SeqTrainingArguments(
    #     **eval_trainer_args,
    #     predict_with_generate=True,
    #     generation_config=GenerationConfig(**gen_kwargs),
    # )

    # # Seq2SeqTrainer is actually just the same as Trainer
    # # (although it uses a different data collator, i.e., explicit prompt/answer separation)
    # # it just allows `predict_with_generate`
    # # allowing us to compute metrics on the generated outputs
    # # no clue why they call this seq2seq...

    # logger.info("=" * 80 + "\n" + "Evaluating model..." + "\n" + "=" * 80)

    # model.eval()

    # eval_trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=eval_trainer_args,
    #     # TODO: use a different collator for test, e.g., more max_len truncation
    #     # w/ left padding?
    #     # removing label part from input_ids
    #     data_collator=generation_collator,
    # )

    # for split, ds in zip(["eval", "test"], [val_dataset, test_dataset]):
    #     if ds is None:
    #         continue
    #     eval_generation(eval_trainer, tokenizer, ds, split, gen_kwargs)
    #     clear_gpu()
