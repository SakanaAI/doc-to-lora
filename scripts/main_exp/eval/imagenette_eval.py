import json
import os
import re
from argparse import Namespace
from difflib import SequenceMatcher

import torch
from datasets import load_dataset
from tqdm import tqdm
from transformers import AutoProcessor, Gemma3ForConditionalGeneration

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.ctx_encoder import PerLayerActivations
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel
from ctx_to_lora.modeling.lora_layer import apply_lora_to_layers
from ctx_to_lora.modeling.lora_merger import combine_lora

CLASS_NAMES = [
    "tench",
    "English springer",
    "cassette player",
    "chain saw",
    "church",
    "French horn",
    "garbage truck",
    "gas pump",
    "golf ball",
    "parachute",
]
CLASS_TO_INT = {name: i for i, name in enumerate(CLASS_NAMES)}
INPUT_TXT = f"What is in this image? Choose exactly one of the following classes: {', '.join(CLASS_NAMES)}. Response with only the correct class without any other text."
RUN_DIR = "train_outputs/runs/Oct16_02-37-04_slurm0-a3nodeset-8_94074_1d62ecb8"


def _normalize_text(text: str) -> str:
    text = re.sub(r"[^a-z0-9\s]", " ", text.lower())
    return " ".join(text.split())


def _normalize_compact(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def _build_alias_map():
    alias_overrides = {
        "english springer spaniel": "English springer",
        "springer spaniel": "English springer",
        "chainsaw": "chain saw",
        "dump truck": "garbage truck",
        "refuse truck": "garbage truck",
        "garbage lorry": "garbage truck",
        "fuel pump": "gas pump",
        "gas station pump": "gas pump",
        "cassette deck": "cassette player",
        "cassette recorder": "cassette player",
        "fish": "tench",
        "tench fish": "tench",
        "french horn instrument": "French horn",
        "golfball": "golf ball",
        "skydiving": "parachute",
        "parachutist": "parachute",
    }

    alias_map = {}

    def register(alias: str, canonical: str):
        alias = _normalize_text(alias)
        if alias:
            alias_map[alias] = canonical
            alias_map[_normalize_compact(alias)] = canonical

    for name in CLASS_NAMES:
        register(name, name)
        register(name.replace(" ", ""), name)
        register(name.replace(" ", "-"), name)

    for alias, canonical in alias_overrides.items():
        register(alias, canonical)

    return alias_map


CLASS_ALIAS_MAP = _build_alias_map()


def pred_to_class_id(pred_txt: str) -> int:
    norm_pred = _normalize_text(pred_txt)
    compact_pred = _normalize_compact(pred_txt)

    for alias, canonical in CLASS_ALIAS_MAP.items():
        if alias and (alias in norm_pred or alias in compact_pred):
            return CLASS_TO_INT[canonical]

    pred_tokens = set(norm_pred.split())
    best_class = None
    best_token_hits = -1
    for name in CLASS_NAMES:
        class_tokens = set(_normalize_text(name).split())
        if class_tokens and class_tokens.issubset(pred_tokens):
            return CLASS_TO_INT[name]
        hits = sum(token in pred_tokens for token in class_tokens)
        if hits > best_token_hits:
            best_token_hits = hits
            best_class = name

    best_ratio = -1.0
    for name in CLASS_NAMES:
        ratio = SequenceMatcher(None, norm_pred, _normalize_text(name)).ratio()
        if ratio > best_ratio:
            best_ratio = ratio
            best_class = name

    return CLASS_TO_INT[best_class]


def load_checkpoint():
    checkpoint_path = f"{RUN_DIR}/checkpoint-80000/pytorch_model.bin"
    state_dict = torch.load(checkpoint_path)

    model = ModulatedPretrainedModel.from_state_dict(
        state_dict,
        train=False,
        base_model_kwargs=dict(attn_implementation="flash_attention_2"),
        use_flash_attn=True,
        use_sequence_packing=False,  # for generation
    )
    tokenizer = get_tokenizer("google/gemma-2-2b-it")
    model.eval()
    return model, tokenizer


def load_ctx_encoder():
    model_id = "google/gemma-3-4b-it"
    ctx_model = Gemma3ForConditionalGeneration.from_pretrained(
        model_id, device_map="auto"
    ).eval()
    ctx_encoder_config = Namespace(ctx_encoder_last_layer=26, keep_lm_head=True)
    ctx_model.language_model = PerLayerActivations(
        ctx_model.language_model, ctx_encoder_config
    )
    processor = AutoProcessor.from_pretrained(model_id)
    return ctx_model, processor


def template_image(img, ctx_processor):
    messages = [
        {
            "role": "system",
            "content": [{"type": "text", "text": ""}],
        },
        {
            "role": "user",
            "content": [
                {"type": "image", "image": img},
            ],
        },
    ]

    inputs = ctx_processor.apply_chat_template(
        messages,
        add_generation_prompt=True,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
    )
    return inputs


@torch.inference_mode
def get_ctx_features(ctx_inputs, ctx_encoder):
    forward_outputs = ctx_encoder(**ctx_inputs, output_hidden_states=True)
    ctx_features = torch.stack(forward_outputs.hidden_states, dim=1)
    return ctx_features


def generate_loras(ctx_inputs, ctx_features):
    generated_loras, _ = model.hypernet.generate_weights(
        ctx_features, attn_mask=torch.ones_like(ctx_inputs["input_ids"])
    )
    generated_loras = combine_lora(
        generated_loras,
        n_chunks=torch.tensor((1,), device=model.device),
        lora_bias=model.hypernet.get_head_bias()
        if model.hypernet.config.use_bias
        else None,
    )
    return generated_loras


def apply_loras(model, generated_loras):
    n_queries = torch.ones(1, dtype=torch.int32, device=model.device)

    apply_lora_to_layers(
        model.base_model,
        model.hypernet.layer_indices,
        generated_loras,
        n_queries,
    )


if __name__ == "__main__":
    model, base_tokenizer = load_checkpoint()
    ctx_encoder, ctx_processor = load_ctx_encoder()
    ds = load_dataset("frgfm/imagenette", "full_size", split="validation")
    # ds = ds.shuffle().select(range(int(0.05 * len(ds))))
    input_ids = base_tokenizer.apply_chat_template(
        [{"role": "user", "content": INPUT_TXT}],
        add_special_tokens=False,
        return_attention_mask=False,
        add_generation_prompt=True,
        return_tensors="pt",
    ).to(model.device)

    preds = []
    pred_txts = []
    corrects = []
    labels = ds["label"]
    for sample in tqdm(ds):
        img = sample["image"]
        ctx_inputs = template_image(img, ctx_processor).to(ctx_encoder.device)
        ctx_features = get_ctx_features(ctx_inputs, ctx_encoder)
        generated_loras = generate_loras(ctx_inputs, ctx_features)
        apply_loras(model, generated_loras)

        model_outputs = model.base_model.generate(
            input_ids, max_new_tokens=256, do_sample=False
        )
        pred_txt = base_tokenizer.decode(
            model_outputs[0][len(input_ids[0]) :], skip_special_tokens=True
        )

        pred_txts.append(pred_txt)
        preds.append(pred_to_class_id(pred_txt))
        is_correct = preds[-1] == labels[len(preds) - 1]
        corrects.append(is_correct)
        print(
            f"GT: {CLASS_NAMES[labels[len(preds) - 1]]}, Pred: {pred_txt} -> {CLASS_NAMES[preds[-1]]}, Correct: {is_correct}"
        )

    acc = sum(corrects) / len(corrects)
    print(f"Final accuracy: {acc:4f}")

    jsonl_path = os.path.join(RUN_DIR, "imagenette_eval.jsonl")
    meta_path = os.path.join(RUN_DIR, "imagenette_eval.meta.json")

    with open(jsonl_path, "w") as f:
        for i, (pred_txt, pred_id, label_id) in enumerate(
            zip(pred_txts, preds, labels)
        ):
            f.write(
                json.dumps(
                    {
                        "index": i,
                        "label": int(label_id),
                        "label_name": CLASS_NAMES[label_id],
                        "pred_text": pred_txt,
                        "pred_class_id": int(pred_id),
                        "pred_class_name": CLASS_NAMES[pred_id],
                        "correct": bool(pred_id == label_id),
                    }
                )
                + "\n"
            )

    meta = {
        "dataset": "frgfm/imagenette",
        "subset": "full_size",
        "split": "validation",
        "run_dir": RUN_DIR,
        "prompt": INPUT_TXT,
        "accuracy": float(acc),
        "num_samples": len(preds),
        "class_names": CLASS_NAMES,
    }
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"Wrote samples to {jsonl_path}")
    print(f"Wrote metadata to {meta_path}")
