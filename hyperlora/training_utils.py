from enum import Enum
import json
import logging
import numpy as np
from transformers import (
    GenerationConfig,
    Trainer,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
)
from transformers.trainer_utils import get_last_checkpoint


TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])

logger = logging.getLogger()


def save_generated_text(samples, output_dir, split):
    with open(f"{output_dir}/{split}_generated_text.jsonl", "w") as f:
        for sample in samples:
            f.write(json.dumps(sample) + "\n")


def decode_test_result(test_dataset, test_result, tokenizer):
    for sample, pred_toks, labels in zip(
        test_dataset, test_result.predictions, test_result.label_ids
    ):
        start_idx = np.argmax(labels != -100, axis=0)
        input_toks = sample["input_ids"][:start_idx]
        gen_toks = pred_toks[start_idx:]
        label_toks = labels[start_idx:]

        input_text = tokenizer.decode(input_toks, skip_special_tokens=True)
        gen_text = tokenizer.decode(gen_toks, skip_special_tokens=True)
        label_text = tokenizer.decode(label_toks, skip_special_tokens=True)
        yield {"input": input_text, "generated": gen_text, "label": label_text}


def train_model(
    model,
    tokenizer,
    training_args,
    train_dataset=None,
    val_dataset=None,
    test_dataset=None,
    data_collator=None,
    compute_metrics=None,
    compute_generation_based_metrics=None,
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

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
    )

    checkpoint = None

    # if training_args.resume_from_checkpoint is not None:
    #     checkpoint = training_args.resume_from_checkpoint
    # elif last_checkpoint is not None:
    #     checkpoint = last_checkpoint

    # print(f"Loaded from the checkpoint: {checkpoint}")

    # TODO: save the best model based on eval loss?
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    metrics = trainer.evaluate()
    trainer.log_metrics("eval", metrics)
    trainer.save_metrics("eval", metrics)
    trainer.save_model()

    ############## Evaluation

    # TODO: generalize gen_kwargs for validation
    gen_kwargs = dict(do_sample=False, max_length=2**13, max_new_tokens=100)
    # pad_token_id=tokenizer.pad_token_id,
    # eos_token_id=?

    # NOTE: could also set kv_cache implementation here
    eval_trainer_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        generation_max_length=2**13,
        generation_config=GenerationConfig(**gen_kwargs),
    )
    eval_trainer_args.eval_strategy = "no"

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
        # train_dataset=train_dataset,
        # eval_dataset=val_dataset,
        # TODO: use a different collator for test, e.g., more max_len
        data_collator=data_collator,
        compute_metrics=compute_generation_based_metrics,
    )
    if val_dataset is not None:
        if isinstance(val_dataset, dict):
            val_dataset = val_dataset["val"]
        eval_result = eval_trainer.predict(
            val_dataset,
            metric_key_prefix="val",
            **gen_kwargs,
        )
        eval_trainer.log_metrics("eval", eval_result.metrics)
        eval_trainer.save_metrics("eval", eval_result.metrics)
        save_generated_text(
            decode_test_result(val_dataset, eval_result, tokenizer),
            split="val",
            output_dir=training_args.output_dir,
        )

    if test_dataset is not None:
        test_result = eval_trainer.predict(test_dataset, **gen_kwargs)

        eval_trainer.log_metrics("test", test_result.metrics)
        eval_trainer.save_metrics("test", test_result.metrics)

        save_generated_text(
            decode_test_result(test_dataset, test_result, tokenizer),
            split="test",
            output_dir=training_args.output_dir,
        )
