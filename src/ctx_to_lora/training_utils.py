import gc
import json
import logging
from collections import defaultdict
from dataclasses import fields
from enum import Enum
from typing import Optional

import datasets
import torch
import numpy as np
from rouge_score import rouge_scorer
from torch.utils.data import RandomSampler, DataLoader
from transformers import (
    GenerationConfig,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    Trainer,
)
from transformers.trainer_utils import (
    get_last_checkpoint,
    has_length,
    seed_worker,
)
from transformers.trainer_pt_utils import LengthGroupedSampler
from transformers.utils import is_datasets_available


TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])

logger = logging.getLogger()


# TODO: refactor to make this Trainer optional
class TrainerWithCustomSampler(Trainer):
    def __init__(self, *args, **kwargs):
        self.train_sampler = kwargs.pop("train_sampler", None)
        self.train_batch_sampler = kwargs.pop("train_batch_sampler", None)
        assert not (
            self.train_sampler and self.train_batch_sampler
        ), "train_sampler and train_batch_sampler cannot be both provided"
        super().__init__(*args, **kwargs)

    # overriding to use custom sampler
    def _get_train_sampler(self) -> Optional[torch.utils.data.Sampler]:
        if self.train_dataset is None or not has_length(self.train_dataset):
            return None

        # Build the sampler.
        if self.args.group_by_length:
            if is_datasets_available() and isinstance(
                self.train_dataset, datasets.Dataset
            ):
                lengths = (
                    self.train_dataset[self.args.length_column_name]
                    if self.args.length_column_name in self.train_dataset.column_names
                    else None
                )
            else:
                lengths = None
            model_input_name = (
                self.processing_class.model_input_names[0]
                if self.processing_class is not None
                else None
            )
            return LengthGroupedSampler(
                self.args.train_batch_size * self.args.gradient_accumulation_steps,
                dataset=self.train_dataset,
                lengths=lengths,
                model_input_name=model_input_name,
            )
        elif self.train_sampler:
            return self.train_sampler
        elif self.train_batch_sampler:
            return self.train_batch_sampler
        else:
            return RandomSampler(self.train_dataset)

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(
                train_dataset, description="training"
            )
        else:
            data_collator = self._get_collator_with_removed_columns(
                data_collator, description="training"
            )

        dataloader_params = {
            # "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            # change "sampler" to "batch_sampler"
            dataloader_params["batch_sampler"] = self._get_train_sampler()
            # dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))


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
    train_sampler=None,
    train_batch_sampler=None,
    # preprocess_logits_for_metrics=None,
    # max_new_tokens=2**13,
    # gen_per_device_eval_batch_size=1,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from the checkpoint: {checkpoint}")

    trainer_cls = Trainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
    )
    if train_batch_sampler or train_sampler:
        trainer_kwargs["train_sampler"] = train_sampler
        trainer_kwargs["train_batch_sampler"] = train_batch_sampler
        trainer_cls = TrainerWithCustomSampler

    trainer = trainer_cls(**trainer_kwargs)

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
