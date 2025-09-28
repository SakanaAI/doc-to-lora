from collections import defaultdict

import torch

# Store activations
activations = defaultdict(list)


def apply_forward_hook(module, name):
    def hook_fn(m, input_args, output_tensor):
        activations[name].append(output_tensor.cpu().detach())

    module.register_forward_hook(hook_fn)


def last_token_pool(
    outputs: dict[str, torch.Tensor], attention_mask: torch.Tensor
) -> torch.Tensor:
    last_hidden_states = (
        outputs["hidden_states"][-1].detach()
        if "hidden_states" in outputs
        else outputs["last_hidden_state"].detach()
    )
    left_padding = attention_mask[:, -1].sum() == attention_mask.shape[0]
    if left_padding:
        return last_hidden_states[:, -1]
    else:
        sequence_lengths = attention_mask.sum(dim=1) - 1
        batch_size = last_hidden_states.shape[0]
        return last_hidden_states[
            torch.arange(batch_size, device=last_hidden_states.device), sequence_lengths
        ]


if __name__ == "__main__":
    from functools import partial

    import torch
    from peft import get_peft_model
    from torch.utils.data import DataLoader
    from tqdm import tqdm

    from ctx_to_lora.data.collator import generation_collator
    from ctx_to_lora.data.processing import get_tokenized_dataset
    from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer
    from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

    model_name_or_path = "google/gemma-2-2b-it"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name_or_path, train=False, requires_grad=False
    )
    ds = get_tokenized_dataset(
        ds_name="squad",
        split="test",
        max_qas_len=-1,
        max_qas_per_sample=1,
        base_model_max_len=8192,
        tokenizer=tokenizer,
        ctx_model_max_len=8192,
        ctx_tokenizer=tokenizer,
        max_ctx_chunk_len=-1,
        min_ctx_chunk_len=-1,
        num_chunk_probs=None,
        max_ctx_chunk_num=None,
        add_ctx_to_chat=False,
        use_kl_loss=True,
        max_new_tokens=0,
        add_self_distill_template=False,
        set_format="pt",
    )
    ds = ds.take(100)

    peft_config = get_lora_config(
        model_name_or_path,
        lora_r=8,
        lora_dropout=0,
        target_modules=["down_proj"],
    )
    peft_config.lora_alpha = 16
    peft_model = get_peft_model(base_model, peft_config)

    checkpoint_path = "train_outputs/runs/Sep06_12-27-02_slurm0-a3nodeset-11_84523_bbcb67ca/checkpoint-40000/pytorch_model.bin"
    model = ModulatedPretrainedModel.from_state_dict(
        state_dict=torch.load(checkpoint_path, weights_only=False),
        train=False,
        base_model_kwargs=dict(attn_implementation="flash_attention_2"),
        use_flash_attn=True,
        use_sequence_packing=False,  # for generation
    )
    dataloader = DataLoader(
        ds,
        batch_size=1,
        shuffle=False,
        sampler=None,
        batch_sampler=None,
        num_workers=0,
        collate_fn=partial(generation_collator, tokenizer=tokenizer),
        pin_memory=False,
        drop_last=False,
        timeout=0,
        persistent_workers=False,
    )

    apply_forward_hook(model.ctx_encoder, "ctx_encoder")
    apply_forward_hook(model.layers, "hypernet_layers")

    for sample_idx, sample in tqdm(enumerate(dataloader)):
        for k, v in sample.items():
            sample[k] = v.to(model.device)
        with torch.no_grad():
            lora_state, _ = model.generate_weights(
                sample["ctx_ids"], sample["ctx_attn_mask"]
            )
