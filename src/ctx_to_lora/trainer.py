import logging

import torch
from torch import nn
from transformers import Trainer
from transformers.trainer_pt_utils import get_parameter_names
from transformers.trainer_utils import IntervalStrategy

from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

logger = logging.getLogger()


def per_ctx_loss_ce(inputs, labels, loss):
    # loss still has masked out elem (0 at labels=-100)
    n_queries_per_ctx = inputs["n_queries"].tolist()

    position_ids = inputs["position_ids"].squeeze(0)
    # account only label positions
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )

    # assumes the input starts with non-assistant tokens
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]

    label_seq_lens = end_label_pos - start_label_pos

    # these stack and split can be optimized but let's keep it simple
    # mean across tokens of each q
    qa_losses = torch.stack(
        [
            loss[start : start + llen].mean()
            for start, llen in zip(start_label_pos, label_seq_lens)
        ]
    )

    # mean across queries of each ctx
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]

    # per-ctx loss
    loss = torch.stack(per_ctx_losses)
    return loss


def per_ctx_loss_kl(inputs, labels, loss):
    # loss is compact (label indices selected)
    n_queries_per_ctx = inputs["n_queries"].tolist()

    position_ids = inputs["position_ids"].squeeze(0)
    # account only label positions
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )
    # assumes the input starts with non-assistant tokens
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]

    label_seq_lens = end_label_pos - start_label_pos

    # find equiv start indices in the already sliced loss vector
    cu_label_seq_lens = torch.cumsum(label_seq_lens, dim=0)
    start_indices = torch.cat(
        (
            torch.tensor([0], device=cu_label_seq_lens.device),
            cu_label_seq_lens[:-1],
        )
    )

    # these stack and split can be optimized but let's keep it simple
    # mean across tokens of each q
    qa_losses = torch.stack(
        [loss[start:end].mean() for start, end in zip(start_indices, cu_label_seq_lens)]
    )

    # mean across queries of each ctx
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]

    # per-ctx loss
    loss = torch.stack(per_ctx_losses)
    return loss


class ModulatedModelTrainer(Trainer):
    # modified from the base Trainer to support per-context average loss
    def get_batch_samples(self, epoch_iterator, num_batches, device):
        # only used with `use_per_ctx_average_loss=True`
        batch_samples = []
        num_items_in_batch = None

        for _ in range(num_batches):
            try:
                batch_samples.append(next(epoch_iterator))
            except StopIteration:
                break

        count_num_items_in_batch = (
            len(batch_samples) > 0
            and "labels" in batch_samples[0]
            and "n_ctx_chunks" in batch_samples[0]
        )

        if count_num_items_in_batch:
            num_items_in_batch = dict()
            num_items_in_batch["ctx"] = torch.tensor(
                sum([batch["n_ctx_chunks"].numel() for batch in batch_samples])
            ).to(device)
            # should we avg over num chunks?
            # num_items_in_batch["ctx"] = sum(
            #     [(batch["ctx_position_ids"] == 0).sum() for batch in batch_samples]
            # )
            num_items_in_batch["labels"] = sum(
                [(batch["labels"].ne(-100)).sum() for batch in batch_samples]
            ).to(device)

        if num_items_in_batch is not None:
            if self.args.average_tokens_across_devices:
                for k in num_items_in_batch:
                    num_items_in_batch[k] = self.accelerator.gather(
                        num_items_in_batch[k]
                    ).sum()

            if torch.is_tensor(num_items_in_batch):
                num_items_in_batch = num_items_in_batch.to(device)

                if self.args.n_gpu > 1 and num_items_in_batch.dim() == 0:
                    # In the DataParallel case, convert the scalar tensor into a 1-dim tensor
                    num_items_in_batch = num_items_in_batch.unsqueeze(0)

        return batch_samples, num_items_in_batch


class DistillationTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        # NOTE: the loss output from this fn will be ***added***
        # meaning that we should always scale the loss wrt `num_items_in_batch`
        # (average over the number of items in the accumulated batch)

        is_train = num_items_in_batch is not None
        labels = inputs.pop("labels", None)
        label_pos = torch.where(labels != -100)
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        if "logprobs_vals" not in inputs:
            return (torch.tensor(0.0), outputs) if return_outputs else torch.tensor(0.0)

        target_logp = inputs.pop("logprobs_vals").squeeze(0)
        indices = inputs.pop("logprobs_indices").squeeze(0)

        assert label_pos[0].shape[0] == target_logp.shape[0], (
            "Label positions and target log probabilities should have the same # tokens."
            f"Got : {label_pos[0].shape[0]=} and {target_logp.shape[0]=}"
        )

        ##### KL loss
        outputs_logits = outputs.logits[label_pos[0], label_pos[1] - 1]  # shift back 1

        logq_full_denom = torch.logsumexp(outputs_logits, dim=-1, keepdim=True)  # (N,1)
        selected_logits = outputs_logits.gather(1, indices)  # (N,K)
        # log softmax at selected indices
        logq_selected = selected_logits - logq_full_denom
        p = target_logp.exp()
        loss = -(p * logq_selected).sum(dim=-1)

        # teacher_logp = torch.full_like(outputs_logits, -torch.inf)
        # teacher_logp.scatter_(1, indices, target_logp)
        # # reduction = "batchmean" if num_items_in_batch is None else "sum"
        # p = teacher_logp.exp()
        # logq = nn.functional.log_softmax(outputs_logits, dim=-1)
        # loss = -torch.sum(p * logq, dim=-1)

        if self.use_per_ctx_average_loss:
            loss = per_ctx_loss_kl(inputs, labels, loss)

        if is_train:
            if self.use_per_ctx_average_loss:
                loss = loss.sum() / num_items_in_batch["ctx"]
            else:
                loss = loss.sum() / num_items_in_batch["labels"]
        else:
            # eval
            loss = loss.mean()

        # if reduction == "batchmean":
        #     loss = loss.mean()
        # elif reduction == "sum":
        #     # loss does not scale with grad acc
        #     # num_items_in_batch does
        #     # this works for both token-avg and ctx-avg
        #     # loss = loss.sum() / num_items_in_batch

        # `num_items_in_batch` is # tokens if `args.use_ctx_average_loss=False``
        # loss = loss.sum() / num_items_in_batch
        #####

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        n_modules = len(gen_loras)
        for module, lora in gen_loras.items():
            l1_norm += lora["A"].abs().sum(0).mean() + lora["B"].abs().sum(0).mean()
        l1_norm /= n_modules
        if is_train:
            # during eval `num_items_in_batch` will be None
            l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm
        #####

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            # compensate `num_items_in_batch` division
            self.log(
                {
                    "kl_loss": loss.item() * scaler,
                    "gen_lora_l1_norm": l1_norm.item() * scaler,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss


def causal_lm_ce_loss(
    logits,
    labels,
    vocab_size: int,
    num_items_in_batch: torch.Tensor | None = None,
    ignore_index: int = -100,
    shift_labels: torch.Tensor | None = None,
    **kwargs,
) -> torch.Tensor:
    # Upcast to float if we need to compute the loss to avoid potential precision issues
    logits = logits.float()

    if shift_labels is None:
        # Shift so that tokens < n predict n
        labels = nn.functional.pad(labels, (0, 1), value=ignore_index)
        shift_labels = labels[..., 1:].contiguous()

    # Flatten the tokens
    logits = logits.view(-1, vocab_size)
    shift_labels = shift_labels.view(-1)
    # Enable model parallelism
    shift_labels = shift_labels.to(logits.device)
    # loss = fixed_cross_entropy(
    #     logits, shift_labels, num_items_in_batch, ignore_index, **kwargs
    # )
    loss = nn.functional.cross_entropy(logits, shift_labels, reduction="none")

    return loss


class CrossEntropyTrainer(ModulatedModelTrainer):
    def __init__(self, *args, **kwargs):
        self.gen_lora_l1_reg_coef = kwargs.pop("gen_lora_l1_reg_coef", 0.0)
        self.use_per_ctx_average_loss = kwargs.pop("use_per_ctx_average_loss", False)
        super().__init__(*args, **kwargs)

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        """
        How the loss is computed by Trainer.
        By default, all models return the loss in the first element.
        Subclass and override for custom behavior.
        """

        is_train = num_items_in_batch is not None
        labels = inputs.pop("labels", None)
        outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)
        # [1, tot_seq_len]
        logits = outputs.logits

        # [tot_seq_len]
        loss = causal_lm_ce_loss(logits, labels, self.model.vocab_size)

        if self.use_per_ctx_average_loss:
            loss = per_ctx_loss_ce(inputs, labels, loss)

        if is_train:
            if self.use_per_ctx_average_loss:
                loss = loss.sum() / num_items_in_batch["ctx"]
            else:
                loss = loss.sum() / num_items_in_batch["labels"]
        else:
            # eval
            loss = loss.mean()

        #####
        # if is_train:
        #     if self.use_per_ctx_average_loss:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch["ctx"]
        #     else:
        #         loss_kwargs["num_items_in_batch"] = num_items_in_batch["labels"]
        # inputs = {**inputs, **loss_kwargs}
        # outputs, (gen_loras, _) = model(**inputs, return_generated_lora=True)

        # # Save past state if it exists
        # if self.args.past_index >= 0:
        #     self._past = outputs[self.args.past_index]

        # if labels is not None:
        #     unwrapped_model = self.accelerator.unwrap_model(model)
        #     if _is_peft_model(unwrapped_model):
        #         model_name = unwrapped_model.base_model.model._get_name()
        #     else:
        #         model_name = unwrapped_model._get_name()
        #     # User-defined compute_loss function
        #     if self.compute_loss_func is not None:
        #         loss = self.compute_loss_func(
        #             outputs, labels, num_items_in_batch=num_items_in_batch["labels"]
        #         )
        #     elif model_name in MODEL_FOR_CAUSAL_LM_MAPPING_NAMES.values():
        #         loss = self.label_smoother(outputs, labels, shift_labels=True)
        #     else:
        #         loss = self.label_smoother(outputs, labels)
        # else:
        #     if isinstance(outputs, dict) and "loss" not in outputs:
        #         raise ValueError(
        #             "The model did not return a loss from the inputs, "
        #             "only the following keys: "
        #             f"{','.join(outputs.keys())}. "
        #             "For reference, the inputs it received are "
        #             f"{','.join(inputs.keys())}."
        #         )
        #     # We don't use .loss here since the model may return tuples instead of ModelOutput.
        #     loss = outputs["loss"] if isinstance(outputs, dict) else outputs[0]
        #####

        ##### unpack gen lora dict and compute regularization loss
        l1_norm = 0
        n_modules = len(gen_loras)
        for module, lora in gen_loras.items():
            l1_norm += lora["A"].abs().sum(0).mean() + lora["B"].abs().sum(0).mean()
        l1_norm /= n_modules
        if is_train:
            # during eval `num_items_in_batch` will be None
            l1_norm /= num_items_in_batch["ctx"]

        total_loss = loss + self.gen_lora_l1_reg_coef * l1_norm
        #####

        scaler = self.args.gradient_accumulation_steps if is_train else 1
        if self.args.average_tokens_across_devices and is_train:
            total_loss *= self.accelerator.num_processes
            scaler *= self.accelerator.num_processes

        # rough estimate of the losses (we only log the values from one step)
        if (self.state.global_step == 1 and self.args.logging_first_step) or (
            self.args.logging_strategy == IntervalStrategy.STEPS
            and self.state.global_step % self.state.logging_steps == 0
        ):
            # compensate `num_items_in_batch` division
            self.log(
                {
                    "ce_loss": loss.item() * scaler,
                    "gen_lora_l1_norm": l1_norm.item() * scaler,
                }
            )

        return (total_loss, outputs) if return_outputs else total_loss


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
        ["scaler", "bias", "layernorm", "rmsnorm", "latents_q"],
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
        logger.info("Training with modulated model.")
        trainer_cls = CrossEntropyTrainer
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
    # if getattr(trainer, "use_per_ctx_average_loss", False):
    #     trainer.get_batch_samples = trainer.get_batch_samples_ctx

    # MONKEY PATCH: remove embedding layers from weight decay
    trainer.get_decay_parameter_names = get_decay_parameter_names

    # Trainer loads the best model after training
    # is done when load_best_model_at_end=True
    train_result = trainer.train(resume_from_checkpoint=checkpoint)
    trainer.log_metrics("train", train_result.metrics)
    trainer.save_model()

    # TODO: add benchmark eval?
    # clear_gpu()
