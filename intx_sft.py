import contextlib
import logging
import os
from copy import deepcopy
from functools import partial
from math import isclose

import numpy as np
import torch
import wandb
from datasets import (
    disable_caching,
    interleave_datasets,
)
from peft import PeftModel
from transformers import (
    AutoConfig,
    set_seed,
)
from transformers.utils import is_liger_kernel_available

from ctx_to_lora.configs import (
    AggregatorArguments,
    ArgumentParser,
    CtxEncoderArguments,
    CtxTrainingArguments,
    DataArguments,
    ExperimentSetup,
    HypernetArguments,
    LoRAArguments,
    ModelArguments,
    TrainingArguments,
)
from ctx_to_lora.data.collator import (  # train_packed_collator,; DefaultDataCollator,
    flatten_if_not_packed,
    train_collator,
)
from ctx_to_lora.data.processing import get_tokenized_dataset, pack
from ctx_to_lora.metrics import (
    Evaluator,
    compute_metrics,
    compute_per_token_acc,
    compute_perplexity,
    compute_prefix_matching,
)
from ctx_to_lora.model_loading import (
    get_lora_config,
    get_model_and_tokenizer,
    get_tokenizer,
)
from ctx_to_lora.modeling.hypernet import (
    ModulatedPretrainedModel,
    get_hypernet_config,
)
from ctx_to_lora.trainer import train_model
from ctx_to_lora.utils import (
    compile_linear,
    extract_cli_args,
    get_run_name,
    log_num_train_params,
    save_yaml,
    setup_logging,
    validate_args,
)

logger = logging.getLogger()

LOCAL_RANK = int(os.getenv("LOCAL_RANK", "0"))


def get_ds_prob(train_ds_len: list[int], total_len: int):
    # if a dataset is smaller than 1%, make it 1%
    probs = [0 for _ in train_ds_len]
    for i, ds_len in enumerate(train_ds_len):
        if ds_len / total_len <= 0.01:
            probs[i] = 0.01
    res_probs = 1 - sum(probs)
    res_total_len = sum([l for l in train_ds_len if (l / total_len) > 0.01])
    for i, ds_len in enumerate(train_ds_len):
        if (ds_len / total_len) > 0.01:
            probs[i] = ds_len / res_total_len * res_probs
    assert isclose(sum(probs), 1.0), (
        f"Probs sum to {sum(probs)} ({probs}), expected 1.0"
    )
    return probs


def main():
    ############ Argument parsing
    parser = ArgumentParser(
        (
            DataArguments,
            CtxTrainingArguments,
            ModelArguments,
            LoRAArguments,
            TrainingArguments,
            HypernetArguments,
            AggregatorArguments,
            CtxEncoderArguments,
        )
    )
    (
        data_args,
        ctx_args,
        model_args,
        lora_args,
        training_args,
        hypernet_args,
        aggregator_args,
        ctx_encoder_args,
    ) = parser.parse()

    # there shouldn't be overlap between args
    validate_args(
        [
            data_args,
            ctx_args,
            model_args,
            lora_args,
            training_args,
            hypernet_args,
            aggregator_args,
            ctx_encoder_args,
        ]
    )

    set_seed(training_args.seed)
    checkpoint_dir = training_args.resume_from_checkpoint
    if checkpoint_dir and not os.path.isdir(checkpoint_dir):
        raise NotADirectoryError(f"Checkpoint{checkpoint_dir} is not a directory")

    # should be the same across processes
    # still possible to have a name crash though
    # logging_dir is just "runs/DATE_TIME_HOSTNAME"
    slurm_job_id = f"_{os.getenv('SLURM_JOB_ID')}" if os.getenv("SLURM_JOB_ID") else ""
    run_name = (
        get_run_name(seed_str=training_args.logging_dir.strip("runs/") + slurm_job_id)
        if not checkpoint_dir
        else checkpoint_dir.strip("/").split("/")[-2]
    )

    output_dir = f"train_outputs/runs/{run_name}"
    setup_logging(output_dir, debug=os.getenv("DEBUG", False))
    logger.debug(f"CMD: {' '.join(os.sys.argv)}")
    cli_args = extract_cli_args(os.sys.argv)
    save_yaml(cli_args, f"{output_dir}/cli_args.yaml")
    if "config" in cli_args:
        config_name = os.path.basename(cli_args["config"]).split(".yaml")[0]
        os.environ["WANDB_TAGS"] = config_name

    run_name = os.path.basename(output_dir)
    training_args.run_name = run_name
    training_args.output_dir = output_dir
    training_args.logging_dir = output_dir

    if (
        training_args.lr_scheduler_type == "cosine_with_min_lr"
        and training_args.lr_scheduler_kwargs is None
    ):
        training_args.lr_scheduler_kwargs = {"min_lr": 1e-7}
    args = {
        **vars(deepcopy(data_args)),
        **vars(deepcopy(ctx_args)),
        **vars(deepcopy(model_args)),
        **vars(deepcopy(lora_args)),
        **vars(deepcopy(training_args)),
        **vars(deepcopy(hypernet_args)),
        **vars(deepcopy(aggregator_args)),
        **vars(deepcopy(ctx_encoder_args)),
    }
    args["deepspeed_plugin"] = None
    logger.debug(f"args: {args}")
    save_yaml(args, f"{output_dir}/args.yaml")

    ############ Model setup
    if not ctx_args.from_pretrained_checkpoint:
        model_name = model_args.model_name_or_path
        model, tokenizer = get_model_and_tokenizer(
            **vars(model_args),
            train=True,
            requires_grad=False,  # ctx_args.exp_setup == ExperimentSetup.FULL_FINETUNE,
            peft_config=get_lora_config(model_name, **vars(lora_args)),
        )
        ctx_name = ctx_encoder_args.ctx_encoder_model_name_or_path
        if ctx_name is not None:
            ctx_encoder_model_config = AutoConfig.from_pretrained(
                ctx_name, trust_remote_code=True
            )
            if "Llama" in ctx_name and "Vision" in ctx_name:
                ctx_encoder_model_config = ctx_encoder_model_config.text_config
            ctx_tokenizer = get_tokenizer(ctx_name, train=True)
        else:
            ctx_name = model.base_model.config.name_or_path
            ctx_encoder_model_config = model.config
            ctx_tokenizer = tokenizer

    if ctx_args.exp_setup == ExperimentSetup.HYPERLORA:
        # TODO: handle only extra_modules case (no target_modules)
        logger.info("Using HyperLoRA")
        if not ctx_args.from_pretrained_checkpoint:
            hypernet_config = get_hypernet_config(
                model,
                ctx_encoder_model_config,
                hypernet_args,
                aggregator_args,
                ctx_encoder_args,
            )
            if ctx_encoder_args.layer_idx is None:
                ctx_encoder_args.layer_idx = (
                    ctx_encoder_model_config.num_hidden_layers // 4
                )
                logger.info(
                    f"Using the first {ctx_encoder_args.layer_idx} layers"
                    " as the context encoder"
                )
            model = ModulatedPretrainedModel(
                model,
                hypernet_config,
                ctx_encoder_args,
                ctx_args.use_kl_loss,
                ctx_args.use_sequence_packing,
            )
            training_args.gen_lora_l1_reg_coef = ctx_args.gen_lora_l1_reg_coef
        else:
            logger.info(
                f"Loading from checkpoint: {ctx_args.from_pretrained_checkpoint}"
            )
            model = ModulatedPretrainedModel.from_state_dict(
                torch.load(ctx_args.from_pretrained_checkpoint, weights_only=False),
                train=True,
                use_flash_attn=model_args.use_flash_attn,
                use_sequence_packing=ctx_args.use_sequence_packing,
            )
            tokenizer = get_tokenizer(model.base_model.config.name_or_path, train=True)
            ctx_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
            if ctx_name is None:
                ctx_name = model.base_model.config.name_or_path
            ctx_tokenizer = get_tokenizer(ctx_name, train=True)
            training_args.gen_lora_l1_reg_coef = ctx_args.gen_lora_l1_reg_coef

        if len([p for p in model.ctx_encoder.parameters() if p.requires_grad]):
            raise ValueError("ctx_encoder contains trainable parameters")
        if len([p for p in model.base_model.parameters() if p.requires_grad]):
            raise ValueError("base model contains trainable parameters")
        # we can't compile the base model bc we will be
        # adding/removing forward hooks during training
        model.hypernet.compile(fullgraph=True, mode="max-autotune")

    else:
        # activate LoRA
        base_model_config = AutoConfig.from_pretrained(
            model_args.model_name_or_path, trust_remote_code=True
        )
        base_model_config.save_pretrained(output_dir)
        logger.info("Using LoRA")
        model.set_adapter("default")
        model = torch.compile(model)

    model.train()
    logger.debug(model)
    log_num_train_params(model)

    ############ Dataset setup
    logger.info("Loading dataset...")

    add_ctx_to_chat = not isinstance(model, ModulatedPretrainedModel)
    tokenizer_kwargs = {"max_length": ctx_args.max_base_len}  # not used
    ctx_tokenizer_kwargs = {"max_length": ctx_args.max_ctx_len}  # not used for now

    _get_tokenized_dataset = partial(
        get_tokenized_dataset,
        base_model_max_len=model.base_model.config.max_position_embeddings,
        tokenizer=tokenizer,
        tokenizer_kwargs=tokenizer_kwargs,
        ctx_model_max_len=model.ctx_encoder.config.max_position_embeddings,
        ctx_tokenizer=ctx_tokenizer,
        ctx_tokenizer_kwargs=ctx_tokenizer_kwargs,
        add_ctx_to_chat=add_ctx_to_chat,
        add_repeat_prompt=ctx_args.add_repeat_prompt,
        repeat_prob=ctx_args.repeat_prob,
        add_negative_prompt=ctx_args.add_negative_prompt,
        use_kl_loss=ctx_args.use_kl_loss,
        set_format=None if ctx_args.use_sequence_packing else "pt",
        # streaming=data_args.streaming,
    )
    tokenized_ds = {"train": dict(), "validation": dict()}
    for split, ds_names in zip(
        ["train", "validation"],
        [data_args.train_ds_names, data_args.val_ds_names],
    ):
        if not ds_names:
            continue
        streaming = (split == "train") and data_args.streaming
        ctx_mgr = (
            training_args.main_process_first()
            if split == "train"
            else contextlib.nullcontext()
        )
        with ctx_mgr:
            # process and tokenize on the main process
            # then other replicas can just load the cached dataset
            # we dont save cache for validation ds
            for ds_name in ds_names:
                ds = _get_tokenized_dataset(ds_name, split, streaming=streaming)

                base_name = os.path.basename(ds_name)
                if ds_name.startswith("self_gen/"):
                    ds_name = "self_gen/" + base_name
                else:
                    ds_name = base_name

                tokenized_ds[split][ds_name] = ds

    train_ds = tokenized_ds["train"]
    logging.info(f"train_ds: {train_ds}")

    val_ds = dict()
    if "validation" in tokenized_ds:
        n_val_samples = data_args.max_val_samples_per_ds
        for ds_name, ds in tokenized_ds["validation"].items():
            if ds is None:
                # take some samples from train_ds
                ds = train_ds[ds_name].take(n_val_samples)
                train_ds[ds_name] = train_ds[ds_name].skip(n_val_samples)

            val_ds[ds_name] = ds
            val_indices = np.random.permutation(len(ds))[:n_val_samples]
            val_ds[ds_name] = val_ds[ds_name].select(val_indices)

    # if data_args.streaming:
    # max_steps = ceil(
    #     sum(DS_LEN[ds] for ds in train_ds)
    #     * training_args.num_train_epochs
    #     / training_args.per_device_train_batch_size
    #     / training_args.gradient_accumulation_steps
    #     / training_args.world_size
    # )
    # training_args.max_steps = max_steps

    # # interleaving streaming datasets
    # # simplify the probs for smaller datasets
    # # slightly upsample those datasets
    # probs = [
    #     0.01 if "fw_qa" not in ds_name else 1 + 0.01 - 0.01 * len(train_ds)
    #     for ds_name in train_ds
    # ]
    # train_ds = interleave_datasets(
    #     list(train_ds.values()),
    #     probabilities=probs,
    #     stopping_strategy="all_exhausted",
    #     seed=training_args.seed,
    # )
    # else:

    train_ds_len = [len(ds) for ds in train_ds.values()]
    total_len = sum(train_ds_len)
    # max_steps = ceil(
    #     total_len
    #     * training_args.num_train_epochs
    #     / training_args.per_device_train_batch_size
    #     / training_args.gradient_accumulation_steps
    #     / training_args.world_size
    # )
    # training_args.max_steps = max_steps
    train_ds = interleave_datasets(
        list(train_ds.values()),
        probabilities=get_ds_prob(train_ds_len, total_len),
        seed=training_args.seed,
        stopping_strategy="all_exhausted",
    )
    logger.info(f"Train dataset length: {len(train_ds)}")

    if ctx_args.use_sequence_packing:
        with training_args.main_process_first():
            logging.info("Packing dataset")
            old_ds_len = len(train_ds)
            train_ds = pack(
                train_ds,
                ctx_args.max_packed_inp_len,
                ctx_args.max_packed_ctx_len,
                max_packed_size=-1,
                num_proc=8,
            )
            logger.info(f"Train dataset length: {len(train_ds)}")
            logger.info(
                f"Avg. # of samples per packed sequence: {old_ds_len / len(train_ds)}"
            )
            logger.info("Setting per_device_train_batch_size to 1")
            training_args.per_device_train_batch_size = 1

    logger.info(f"train_ds: {train_ds}")
    logger.info(f"val_ds: {val_ds}")

    collator = (
        flatten_if_not_packed
        # DefaultDataCollator(return_tensors="pt")
        if ctx_args.use_sequence_packing
        else partial(train_collator, tokenizer=tokenizer)
    )

    # TODO: use SFTTrainer instead? https://huggingface.co/docs/trl/en/sft_trainer

    # HACK [local patch]: deepspeed model loading problem (for resume training)
    # see https://github.com/microsoft/DeepSpeed/pull/6626/files
    # /home/rujikorn_sakana_ai/.conda/envs/ctx-to-lora/lib/python3.10/site-packages/deepspeed/runtime/engine.py
    # HACK: [local path]: make liger kernel works with gemma3
    # see .venv/lib/python3.10/site-packages/liger_kernel/transformers/monkey_patch.py
    # L810:814 + L781
    if training_args.use_liger_kernel and is_liger_kernel_available():
        from liger_kernel.transformers import _apply_liger_kernel_to_instance

        if isinstance(model, ModulatedPretrainedModel):
            logger.info("Applying liger-kernel to ModulatedPretrainedModel")
            if isinstance(model.base_model, PeftModel):
                base_model = model.base_model.base_model
            else:
                base_model = model.base_model

            if ctx_name is not None:
                logger.info("Applying liger-kernel to ctx_encoder_model")
                ctx_base_model = model.ctx_encoder.base_model
                _apply_liger_kernel_to_instance(model=ctx_base_model)
                compile_linear(ctx_base_model)
                # model.ctx_encoder.compile(fullgraph=True)

        elif isinstance(model, PeftModel):
            logger.info("Applying liger-kernel to PeftModel")
            base_model = model.base_model

        _apply_liger_kernel_to_instance(model=base_model.model)
        # compile_linear(base_model)
        base_model.compile(fullgraph=True, mode="max-autotune")
        base_model.loss_function = torch.compile(
            base_model.loss_function, fullgraph=True, mode="max-autotune"
        )

    if LOCAL_RANK == 0:
        wandb.init(
            project=os.getenv("WANDB_PROJECT"),
            name=run_name,
            group=run_name,
            config=args,
            tags=os.getenv("WANDB_TAGS").split(","),
            notes=ctx_args.notes,
            resume="allow",
        )
    else:
        wandb.init(mode="disabled")

    train_model(
        model,
        training_args,
        train_ds,
        val_ds,
        collator,
        compute_metrics=partial(
            compute_metrics,
            evaluator=Evaluator(
                [compute_per_token_acc, compute_prefix_matching, compute_perplexity]
            ),
        ),
    )
    logger.info(f"Training run finished and saved to {output_dir}")


if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["TOKENIZERS_PARALLELISM"] = "true"
    os.environ["WANDB_DIR"] = ".wandb/"
    os.environ["WANDB_PROJECT"] = "ctx_to_lora"
    os.environ["WANDB_WATCH"] = ""  # "all"
    os.environ["WANDB_CONSOLE"] = "off"
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    # os.environ["KMP_AFFINITY"] = "disabled"  # fixing iterable dataset stuck
    os.environ["OMP_NUM_THREADS"] = "23"
    # os.environ["HF_DATASETS_IN_MEMORY_MAX_SIZE"] = "137438953472"  # 128 TB
    torch._dynamo.config.capture_scalar_outputs = True

    # DOES THIS CAUSE ACCURACY DEGRADATION?
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    if os.getenv("DEBUG", False):
        disable_caching()
    main()
