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
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
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
        if use_kl:
            labels = inputs.pop("labels", None)
            label_pos = torch.where(labels != -100)
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        if not use_kl:
            loss = outputs["loss"]

        else:
            target_logp = inputs.pop("logprobs_vals").squeeze(0)
            indices = inputs.pop("logprobs_indices").squeeze(0)

            assert label_pos[0].shape[0] == target_logp.shape[0], (
                "Label positions and target log probabilities should have the same # tokens."
                f"Got : {label_pos[0].shape[0]=} and {target_logp.shape[0]=}"
            )

            ##### KL loss
            outputs_logits = outputs.logits[
                label_pos[0], label_pos[1] - 1
            ]  # shift back 1

            teacher_logp = torch.full_like(outputs_logits, -torch.inf)
            teacher_logp.scatter_(1, indices, target_logp)
            reduction = "batchmean" if num_items_in_batch is None else "sum"

            p = teacher_logp.exp()
            logq = nn.functional.log_softmax(outputs_logits, dim=-1)
            loss = -torch.sum(p * logq, dim=-1)

            if self.use_per_ctx_average_loss:
                n_queries_per_ctx = inputs["n_queries"].tolist()

                position_ids = inputs["position_ids"].squeeze(0)
                # account only label positions
                label_mask = labels.squeeze(0) != -100
                label_pos_ids = label_mask * position_ids
                label_pos_ids_diff = label_pos_ids.diff(
                    append=torch.tensor([0], device=position_ids.device)
                )
                start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
                end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]

                label_seq_lens = end_label_pos - start_label_pos
                cu_label_seq_lens = torch.cumsum(label_seq_lens, dim=0)
                start_indices = torch.cat(
                    (
                        torch.tensor([0], device=cu_label_seq_lens.device),
                        cu_label_seq_lens[:-1],
                    )
                )

                # these stack and split can be optimized but let's keep it simple
                # Calculate mean loss for each QA pair
                qa_losses = torch.stack(
                    [
                        loss[start:end].mean()
                        for start, end in zip(start_indices, cu_label_seq_lens)
                    ]
                )

                # Split losses by context and calculate per-context average loss
                per_ctx_losses = [
                    ctx.mean() for ctx in torch.split(qa_losses, n_queries_per_ctx)
                ]

                # Final loss is the average of per-context losses
                loss = torch.stack(per_ctx_losses).mean()

            elif reduction == "batchmean":
                loss = loss.mean()
            elif reduction == "sum" and num_items_in_batch is not None:
                loss = loss.sum() / num_items_in_batch
            #####

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        n_modules = len(gen_loras)
        for module, lora in gen_loras.items():
            l1_norm += lora["A"].abs().mean() + lora["B"].abs().mean()
        l1_norm /= n_modules
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
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
        # Save past state if it exists
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
        trainer_kwargs["use_per_ctx_average_loss"] = (
            training_args.use_per_ctx_average_loss
        )
        del training_args.gen_lora_l1_reg_coef
        del training_args.use_per_ctx_average_loss

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
