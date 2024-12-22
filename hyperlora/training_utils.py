import math
import os
import random
from enum import Enum, auto

import torch
from transformers import Seq2SeqTrainer, Trainer
from transformers.trainer_utils import get_last_checkpoint

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])


def train_model(
    model,
    train_dataset,
    eval_dataset,
    training_args,
    data_collator=None,
    compute_metrics=None,
):

    last_checkpoint = None
    if (
        os.path.isdir(training_args.output_dir)
        and not training_args.overwrite_output_dir
    ):
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir})"
                " already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif (
            last_checkpoint is not None and training_args.resume_from_checkpoint is None
        ):
            print(
                f"Checkpoint detected, resuming training at {last_checkpoint}. "
                "To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    if (
        max(
            training_args.per_device_train_batch_size,
            training_args.per_device_eval_batch_size,
        )
        == 1
    ):
        data_collator = None

    # print training_args at local_rank 0
    local_rank = int(os.getenv("LOCAL_RANK", "0"))
    if local_rank == 0:
        print(training_args)

    # Seq2SeqTrainer is actually just the same as Trainer
    # (although it uses a different data collator, i.e., explicit prompt/answer separation)
    # it just allows `predict_with_generate`
    # allowing us to compute metrics on the generated outputs
    # no clue why they call this seq2seq...
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    checkpoint = None

    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
    elif last_checkpoint is not None:
        checkpoint = last_checkpoint

    print(f"Loaded from the checkpoint: {checkpoint}")

    # TODO: save the best model based on eval loss?
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model()
