import gc
import logging
from enum import Enum

import torch
from transformers import Trainer
from transformers.trainer import _is_peft_model
from transformers.trainer_utils import IntervalStrategy
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES

from ctx_to_lora.modeling_utils import ModulatedPretrainedModel

TRAINING_TASK = Enum("TRAINING_TASK", ["CAUSAL_LM", "COMPLETION"])

logger = logging.getLogger()


class ModulatedModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        super().__init__(*args, **kwargs)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        """
        How the loss is computed by Trainer. By default, all models return the loss in the first element.

        Subclass and override for custom behavior.
        """
        if (
            self.label_smoother is not None or self.compute_loss_func is not None
        ) and "labels" in inputs:
            labels = inputs.pop("labels")
        else:
            labels = None
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
        # Save past state if it exists
        # TODO: this needs to be fixed and made cleaner later.
        if self.args.past_index >= 0:
            self._past = outputs[self.args.past_index]

        if labels is not None:
            unwrapped_model = self.accelerator.unwrap_model(model)
            if _is_peft_model(unwrapped_model):
                model_name = unwrapped_model.base_model.model._get_name()
            else:
                model_name = unwrapped_model._get_name()
            # User-defined compute_loss function
            if self.compute_loss_func is not None:
                loss = self.compute_loss_func(
                    outputs, labels, num_items_in_batch=num_items_in_batch
                )
            elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
                loss = self.label_smoother(outputs, labels, shift_labels=True)
            else:
                loss = self.label_smoother(outputs, labels)
        else:
            if isinstance(outputs, dict) and "loss" not in outputs:
                raise ValueError(
                    "The model did not return a loss from the inputs, only the following keys: "
                    f"{','.join(outputs.keys())}. For reference, the inputs it received are {','.join(inputs.keys())}."
                )
            # We don't use .loss here since the model may return tuples instead of ModelOutput.
            loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        for module, lora in gen_loras.items():
            l1_norm += lora["A"].abs().mean() + lora["B"].abs().mean()
        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            self.log({"ce_loss": loss.item(), "gen_lora_l1_norm": l1_norm.item()})
        loss += self.gen_lora_l1_reg_coef * l1_norm
        #####

        if self.args.average_tokens_across_devices and self.model_accepts_loss_kwargs:
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


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
    # test_dataset=None,
    train_collator=None,
    # generation_collator=None,
    compute_metrics=None,
    train_sampler=None,
    # preprocess_logits_for_metrics=None,
    # max_new_tokens=2**13,
    # gen_per_device_eval_batch_size=1,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from the checkpoint: {checkpoint}")

    is_modulated_model = isinstance(model, ModulatedPretrainedModel)
    trainer_cls = Trainer  # if not is_modulated_model else ModulatedModelTrainer
    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
    )
    # if is_modulated_model:
    #     logger.info(f"Training with modulated model. Using CustomTrainer.")
    #     trainer_kwargs["gen_lora_l1_reg_coef"] = training_args.gen_lora_l1_reg_coef
    #     del training_args.gen_lora_l1_reg_coef

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
