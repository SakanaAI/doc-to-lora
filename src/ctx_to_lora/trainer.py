import logging

import torch
from torch import nn
from transformers import Trainer
from transformers.models.auto.modeling_auto import MODEL_FOR_CAUSAL_LM_MAPPING_NAMES
from transformers.trainer import _is_peft_model
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import IntervalStrategy

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

logger = logging.getLogger()


class DistillationTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.distillation_temperature = kwargs.pop("distillation_temperature", 1.0)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if self.model_accepts_loss_kwargs:
            loss_kwargs = {}
            if num_items_in_batch is not None:
                loss_kwargs["num_items_in_batch"] = num_items_in_batch
            inputs = {**inputs, **loss_kwargs}
        use_kl = "logprobs_vals" in inputs
        # TODO: handle when val data doesnt have logprobs
        if use_kl:
            labels = inputs.pop("labels", None)
            label_pos = torch.where(labels != -100)
        # TODO: add efficient logits indexing?
        # https://github.com/huggingface/trl/pull/2773
        # + use `num_logits_to_keep` to limit the number of logits
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        if not use_kl:
            loss = outputs["loss"]

        else:
            target_logp = inputs.pop("logprobs_vals")
            indices = inputs.pop("logprobs_indices")

            # packed
            if isinstance(target_logp, torch.Tensor):
                target_logp = target_logp.squeeze(0)
                indices = indices.squeeze(0)
            # non-packed max_qas_per_sample=1
            elif isinstance(target_logp[0], torch.Tensor):
                target_logp = torch.cat([x[0] for x in target_logp], dim=0)
                indices = torch.cat([x[0] for x in indices], dim=0)
            # non-packed max_qas_per_sample>1
            else:
                # convert to tensor
                raise NotImplementedError(
                    "Non-packed data with max_qas_per_sample > 1 is not supported yet."
                )

            assert label_pos[0].shape[0] == target_logp.shape[0], (
                "Label positions and target log probabilities should have the same # tokens."
                f"Got : {label_pos[0].shape[0]=} and {target_logp.shape[0]=}"
            )

            # TODO: per-ctx average loss?
            ##### KL loss
            outputs_logits = outputs.logits[
                label_pos[0], label_pos[1] - 1
            ]  # shift back 1

            teacher_logp = torch.full_like(outputs_logits, -torch.inf)
            teacher_logp.scatter_(1, indices, target_logp)
            reduction = "batchmean" if num_items_in_batch is None else "sum"

            p = teacher_logp.exp()
            logq = nn.functional.log_softmax(outputs_logits, dim=-1)
            loss = -torch.sum(p * logq, dim=-1) * (self.distillation_temperature**2)
            if reduction == "batchmean":
                loss = loss.mean()
            elif reduction == "sum" and num_items_in_batch is not None:
                loss = loss.sum() / num_items_in_batch
            #####

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        for module, lora in gen_loras.items():
            l1_norm += lora["A"].abs().mean() + lora["B"].abs().mean()
        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            loss_field = "kl_loss" if use_kl else "ce_loss"
            self.log({loss_field: loss.item(), "gen_lora_l1_norm": l1_norm.item()})
        loss += self.gen_lora_l1_reg_coef * l1_norm
        #####

        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


class ModulatedModelTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.
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
        # TODO: add efficient logits indexing?
        # https://github.com/huggingface/trl/pull/2773
        # + use `num_logits_to_keep` to limit the number of logits
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
                    "The model did not return a loss from the inputs, "
                    "only the following keys: "
                    f"{','.join(outputs.keys())}. "
                    "For reference, the inputs it received are "
                    f"{','.join(inputs.keys())}."
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

        if (
            self.args.average_tokens_across_devices
            and self.model_accepts_loss_kwargs
            and num_items_in_batch is not None
        ):
            loss *= self.accelerator.num_processes

        return (loss, outputs) if return_outputs else loss


def get_decay_parameter_names(model) -> list[str]:
    """
    Get all parameter names that weight decay will be applied to.

    This function filters out parameters in two ways:
    1. By layer type (nn.Embedding)
    2. By parameter name patterns (containing 'bias', 'layernorm', 'rmsnorm'
       or 'latents_q' [perceiver's latent queries]).
    """
    decay_parameters = get_parameter_names(
        model,
        [nn.Embedding, nn.LayerNorm],
        ["bias", "layernorm", "rmsnorm", "latents_q"],
    )
    return decay_parameters


def train_model(
    model,
    training_args,
    train_dataset=None,
    val_dataset=None,
    train_collator=None,
    compute_metrics=None,
):
    checkpoint = None
    if training_args.resume_from_checkpoint is not None:
        checkpoint = training_args.resume_from_checkpoint
        logger.info(f"Resuming from the checkpoint: {checkpoint}")

    trainer_kwargs = dict(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=val_dataset,
        data_collator=train_collator,
        compute_metrics=compute_metrics,
    )

    is_modulated_model = isinstance(model, ModulatedPretrainedModel)
    trainer_cls = Trainer
    if is_modulated_model:
        logger.info("Training with modulated model. Using ModulatedModelTrainer.")
        trainer_cls = ModulatedModelTrainer
        trainer_kwargs["gen_lora_l1_reg_coef"] = training_args.gen_lora_l1_reg_coef
        del training_args.gen_lora_l1_reg_coef

        if training_args.use_kl_loss:
            logger.info("Training with distillation loss. Using DistillationTrainer.")
            trainer_cls = DistillationTrainer
            del training_args.use_kl_loss

    if training_args.auto_find_batch_size:
        # set the batch size to some high number
        # which will be lowered by the Trainer
        training_args.per_device_train_batch_size = 128

    trainer = trainer_cls(**trainer_kwargs)
    # MONKEY PATCH: remove embedding layers from weight decay
    trainer.get_decay_parameter_names = get_decay_parameter_names

    # Trainer loads the best model after training
    # is done when load_best_model_at_end=True
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_model()

    # TODO: add benchmark eval?
    # clear_gpu()
