import logging
import random
from typing import Any

import torch
from jaxtyping import Integer
from peft import PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from torch import Tensor, nn
from transformers import PreTrainedModel
from transformers.modeling_outputs import ModelOutput

from ctx_to_lora.data.definitions import CTX_AFFIXES
from ctx_to_lora.data.q_generation_prompts import (
    PROMPT_TEMPLATE,
    SYSTEM_PROMPT,
    get_creative_seed_prompt,
    get_generic_seed_prompt,
    get_question_seed_prompt,
    get_use_case_seed_prompt,
)
from ctx_to_lora.data.self_gen_template import SELF_QA_INTX
from ctx_to_lora.utils import log_num_train_params

logger = logging.getLogger()


def build_messages(
    context: str,
    question_weight: float,
    use_case_weight: float,
    creative_weight: float,
    generic_weight: float,
) -> list[dict]:
    # Create list of instruction functions with their weights
    instruction_options = [
        ("question", get_question_seed_prompt, question_weight),
        ("use_case", get_use_case_seed_prompt, use_case_weight),
        ("creative", get_creative_seed_prompt, creative_weight),
        ("generic", get_generic_seed_prompt, generic_weight),
    ]

    q_types, functions, weights = zip(*instruction_options)

    chosen_idx = random.choices(range(len(instruction_options)), weights=weights)[0]

    function = functions[chosen_idx]

    custom_instruction = function()
    messages = [
        {"role": "system", "content": SYSTEM_PROMPT},
        {
            "role": "user",
            "content": PROMPT_TEMPLATE.format(
                context=context, instruction=custom_instruction
            ),
        },
    ]
    return messages


def get_shifted_label_pos(labels):
    pos = torch.where(labels != -100)
    # (batch_idx, token_idx)
    return (pos[0], pos[1] - 1)


def logits_at_positions(outputs: ModelOutput, pos) -> Tensor:
    logits = outputs.logits
    return logits[pos[0], pos[1]]


def ctx_inp_split(
    ctx_inp_ids, ctx_inp_sep_seq, pad_token_id, prefix_tokens=None, padding_side="right"
):
    # Split each row in ctx_inp_ids at the first occurrence of ctx_inp_sep_seq
    # Return the part after the separator for each row
    batch_size = ctx_inp_ids.size(0)
    sep_len = ctx_inp_sep_seq.size(0)
    out_inp = []
    out_ctx = []
    for i in range(batch_size):
        row = ctx_inp_ids[i]
        # Find where the separator starts
        for j in range(row.size(0) - sep_len + 1):
            if torch.equal(row[j : j + sep_len], ctx_inp_sep_seq):
                out_ctx.append(row[:j])
                if prefix_tokens is not None:
                    out_inp.append(
                        torch.cat([prefix_tokens, row[j + sep_len :]], axis=-1)
                    )
                else:
                    out_inp.append(row[j + sep_len :])
                break
        else:
            # If separator not found
            raise ValueError(f"Separator sequence not found in row {i}")
    out_inp = torch.nn.utils.rnn.pad_sequence(
        out_inp, batch_first=True, padding_value=pad_token_id, padding_side=padding_side
    )
    out_ctx = torch.nn.utils.rnn.pad_sequence(
        out_ctx, batch_first=True, padding_value=pad_token_id, padding_side=padding_side
    )
    return out_ctx, out_inp


def get_peft_layers(model, peft_config):
    out = []
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(module.base_layer, nn.Linear), (
            "all modules should be a leave module that is Linear layer"
        )

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules
        out.append(module)

    return out


class CtxDistillModel(nn.Module):
    def __init__(
        self,
        base_model: PeftModel,
        prefix_tokens: Integer[Tensor, "n"],
        pad_token_id: int,
        update_iterations: int,
        q_model: PreTrainedModel | None = None,
        num_gen_q: int | None = None,
        tokenizer=None,
    ):
        super().__init__()
        self.register_module("base_model", base_model)
        self.register_module("q_model", q_model)
        if q_model is not None:
            assert num_gen_q is not None, (
                "num_gen_q must be provided if q_model is used"
            )
            self.num_gen_q = num_gen_q
        self.register_buffer("prefix_tokens", prefix_tokens)
        self.tokenizer = tokenizer
        self.pad_token_id = pad_token_id
        self.update_iterations = update_iterations
        self.device = base_model.device
        self.to(self.device)

        self.peft_config = base_model.peft_config["default"]
        self.adapter_name = "default"
        self.base_model.set_adapter("default")
        for layer in get_peft_layers(self.base_model, self.peft_config):
            for name, p in layer.named_parameters():
                if "lora_A" in name or "lora_B" in name:
                    p.requires_grad = True
        log_num_train_params(self.base_model)
        self._init_optim()

    @property
    def generation_config(self):
        return self.base_model.generation_config

    def _init_optim(self):
        self.optimizer = torch.optim.AdamW(
            [
                p
                for l in get_peft_layers(self.base_model, self.peft_config)
                for p in l.parameters()
                if p.requires_grad
            ],
            lr=1e-4,
        )

    def reset_lora(self):
        print("Resetiing LoRA")
        for layer in get_peft_layers(self.base_model, self.peft_config):
            layer.reset_lora_parameters(self.adapter_name, init_lora_weights=True)
        self._init_optim()

    @torch.enable_grad()
    def _distill_context(
        self,
        ctx_inp_res_ids: Integer[Tensor, "bs ctx_inp_length"],
        ctx_inp_res_attention_mask: Integer[Tensor, "bs ctx_inp_length"],
        teacher_labels: Integer[Tensor, "bs ctx_inp_length"],
        inp_res_ids: Integer[Tensor, "bs inp_length"],
        inp_res_attention_mask: Integer[Tensor, "bs inp_length"],
        student_labels: Integer[Tensor, "bs inp_length"],
    ):
        # Implements KD-style loss by computing teacher (with context) and student (no context) log-probs locally.

        teacher_shifted_label_pos = get_shifted_label_pos(teacher_labels)
        student_shifted_label_pos = get_shifted_label_pos(student_labels)

        with torch.no_grad():
            teacher_outputs = self.base_model(
                ctx_inp_res_ids, attention_mask=ctx_inp_res_attention_mask
            )
            teacher_logits = logits_at_positions(
                teacher_outputs, teacher_shifted_label_pos
            )
            # Use top-k teacher probabilities only
            K = 16
            topk_vals, topk_idx = teacher_logits.topk(K, dim=-1)
            teacher_denom = torch.logsumexp(
                teacher_logits.float(), dim=-1, keepdim=True
            )
            teacher_p = (topk_vals - teacher_denom).exp().detach()  # [N, K]

        # Optimization loop: match student to teacher using selected indices
        was_training = self.training
        self.train()

        print(f"Starting context distillation for {self.update_iterations} iterations")

        for iteration in range(self.update_iterations):
            self.optimizer.zero_grad()
            student_outputs = self.base_model(
                inp_res_ids, attention_mask=inp_res_attention_mask
            )
            student_logits = logits_at_positions(
                student_outputs, student_shifted_label_pos
            )
            student_denom = torch.logsumexp(
                student_logits.float(), dim=-1, keepdim=True
            )
            selected_student_logits = student_logits.gather(-1, topk_idx)
            student_logq = selected_student_logits - student_denom  # [N, K]
            token_losses = -(teacher_p * student_logq).sum(dim=-1)  # [N]
            loss = token_losses.mean()

            # Log metrics for this iteration
            print(
                f"Iteration {iteration + 1}/{self.update_iterations}: "
                f"Loss={loss.item():.4f}"
            )

            loss.backward()
            self.optimizer.step()

        if not was_training:
            self.eval()

    def generate_questions(self, *args, **kwargs):
        return self.q_model.generate(*args, **kwargs)

    def teacher_generate(self, *args, **kwargs):
        # rename for separate timing
        return self.base_model.generate(*args, **kwargs)

    def student_generate(self, *args, **kwargs):
        # rename for separate timing
        return self.base_model.generate(*args, **kwargs)

    def generate(
        self,
        # ctx_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        # ctx_attn_mask: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        # ctx_position_ids: Integer[Tensor, "n_chunks ctx_length"] | None = None,
        # n_ctx_chunks: Integer[Tensor, "n_ctx"] | None = None,
        # n_queries: Integer[Tensor, "n_ctx"] | None = None,
        *model_inputs_args: Any,
        ctx_inp_sep_seq: Integer[Tensor, "l"],
        reset: bool,
        **model_inputs_kwargs: dict[str, Any],
    ):
        # where to get the questions???
        # TODO: contex indep variant
        # self.generate_questions(ctx_ids)

        # update peft module with CD (if labels provided)

        if reset:
            self.reset_lora()

        # teacher tokens
        if model_inputs_args:
            ctx_inp_ids = model_inputs_args[0]
        else:
            ctx_inp_ids = model_inputs_kwargs.pop("input_ids")
        _, orig_inp_ids = ctx_inp_split(
            ctx_inp_ids,
            ctx_inp_sep_seq,
            self.pad_token_id,
            self.prefix_tokens,
            padding_side="left",
        )
        ctx_inp_attention_mask = model_inputs_kwargs.pop("attention_mask")

        if self.q_model is not None:
            # Extract context-only portion after separator (remove prefix tokens from first row)
            ctx_ids_full, _ = ctx_inp_split(
                ctx_inp_ids, ctx_inp_sep_seq, self.pad_token_id
            )  # [bs, var_len]
            ctx_ids = ctx_ids_full[0, len(self.prefix_tokens) :]
            ctx_txt = self.tokenizer.decode(ctx_ids, skip_special_tokens=True)
            # Build multiple instruction variants
            messages_list = [
                build_messages(ctx_txt, 1, 1, 1, 1) for _ in range(self.num_gen_q)
            ]
            q_inputs = self.tokenizer.apply_chat_template(
                messages_list,
                tokenize=True,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            q_inputs = {k: v.to(self.device) for k, v in q_inputs.items()}

            with torch.no_grad():
                question_outputs = self.generate_questions(
                    input_ids=q_inputs["input_ids"],
                    attention_mask=q_inputs.get("attention_mask", None),
                    max_new_tokens=256,
                    do_sample=True,
                    top_p=0.95,
                    temperature=1.0,  # high temp for diverse questions
                )
            # Slice off the prompt portion
            gen_only = question_outputs[:, q_inputs["input_ids"].shape[-1] :]
            questions = self.tokenizer.batch_decode(gen_only, skip_special_tokens=True)
            questions = [q.split("Message:")[-1].strip() for q in questions]

            ctx_inp_messages = [
                [{"role": "user", "content": f"{ctx_txt}\n\n{SELF_QA_INTX}\n\n{q}"}]
                for q in questions
            ]
            encoded_ctx_inp = self.tokenizer.apply_chat_template(
                ctx_inp_messages,
                tokenize=True,
                add_special_tokens=False,
                padding=True,
                truncation=False,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )
            encoded_ctx_inp = {k: v.to(self.device) for k, v in encoded_ctx_inp.items()}
            ctx_inp_ids = encoded_ctx_inp["input_ids"]
            ctx_inp_attention_mask = encoded_ctx_inp["attention_mask"]

        # TODO: check labels + loss calculation + padding

        # sample responses first
        ctx_inp_res_ids = self.teacher_generate(
            ctx_inp_ids,
            attention_mask=ctx_inp_attention_mask,
            **model_inputs_kwargs,
        )
        ctx_inp_res_attention_mask = torch.where(
            ctx_inp_res_ids != self.pad_token_id, 1, 0
        ).long()
        ctx_inp_res_txt = self.tokenizer.batch_decode(ctx_inp_res_ids)

        bs = ctx_inp_ids.shape[0]
        res_len = ctx_inp_res_ids.shape[-1] - ctx_inp_ids.shape[-1]
        res_ids = ctx_inp_res_ids[:, -res_len:]  # correct

        pads = torch.full_like(ctx_inp_ids, self.pad_token_id)
        teacher_labels = torch.cat([pads, res_ids], dim=-1)
        teacher_labels = torch.where(
            teacher_labels != self.pad_token_id, teacher_labels, -100
        )

        # student tokens
        _, inp_res_ids = ctx_inp_split(
            ctx_inp_res_ids,
            ctx_inp_sep_seq,
            self.pad_token_id,
            self.prefix_tokens,
            padding_side="left",
        )
        inp_res_attention_mask = torch.where(
            inp_res_ids != self.pad_token_id, 1, 0
        ).long()

        # _, inp_ids = ctx_inp_split(
        #     ctx_inp_ids,
        #     ctx_inp_sep_seq.to(ctx_inp_res_ids.device),
        #     self.pad_token_id,
        #     self.prefix_tokens,
        # )

        student_labels = inp_res_ids.clone()
        student_labels[:, :-res_len] = -100
        student_labels = torch.where(
            student_labels != self.pad_token_id, student_labels, -100
        )

        self._distill_context(
            ctx_inp_res_ids,
            ctx_inp_res_attention_mask,
            teacher_labels,
            inp_res_ids,
            inp_res_attention_mask,
            student_labels,
        )

        # _, inp_ids = ctx_inp_split(
        #     ctx_inp_ids,
        #     ctx_inp_sep_seq.to(ctx_inp_res_ids.device),
        #     self.pad_token_id,
        #     self.prefix_tokens,
        #     padding_side="left",
        # )

        # # inp_ids = torch.cat([self.prefix_tokens.expand(bs, -1), inp_ids], dim=-1)
        # # inp_ids = inp_res_ids[:, :-res_len]
        # print(self.tokenizer.batch_decode(inp_ids))
        inp_attention_mask = torch.where(orig_inp_ids != self.pad_token_id, 1, 0).long()
        model_inputs_kwargs.pop("attention_mask", None)
        model_inputs_kwargs.pop("input_ids", None)
        model_outputs = self.student_generate(
            orig_inp_ids, attention_mask=inp_attention_mask, **model_inputs_kwargs
        )
        return model_outputs


if __name__ == "__main__":
    from ctx_to_lora.data.processing import load_and_process_dataset
    from ctx_to_lora.model_loading import get_lora_config, get_model_and_tokenizer

    model_name = "google/gemma-2-2b-it"
    peft_config = get_lora_config(
        model_name, r=8, target_modules=["down_proj"], lora_dropout=0.0
    )
    peft_config.lora_alpha = 16
    model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=False,
        requires_grad=False,
        peft_config=peft_config,
    )

    ds = load_and_process_dataset("pwc", split="train", num_proc=8)
    ctx = ds[0]["context"]
    inp = ds[1]["prompts"][0]
    # Build a simple context/input pair separated by a unique token sequence
    sep_text = SELF_QA_INTX
    # ctx = "# Provided Information\nMy name is Tan."
    # ctx = "# Provided Info"
    # inp = "What is my name? Reply with my name only. No explanation."
    prompt = f"{ctx}\n\n{sep_text}\n\n{inp}"
    messages = [{"role": "user", "content": prompt}]
    encoded = tokenizer.apply_chat_template(
        messages, return_tensors="pt", return_dict=True
    )
    # encoded = tokenizer(prompt, return_tensors="pt")
    encoded = {k: v.to(model.device) for k, v in encoded.items()}

    prefix_tokens = CTX_AFFIXES[model_name]["prefix"]
    prefix_tokens = torch.tensor(prefix_tokens, dtype=torch.long)

    cd_model = CtxDistillModel(
        base_model=model,
        prefix_tokens=prefix_tokens,
        pad_token_id=tokenizer.pad_token_id,
        update_iterations=200,
        q_model=model,
        num_gen_q=20,
        tokenizer=tokenizer,
    )

    sep_ids = (
        tokenizer(sep_text.strip("\n"), add_special_tokens=False, return_tensors="pt")
        .input_ids[0]
        .to(model.device)
    )

    with torch.no_grad():
        for _ in range(1):
            base_model_res = model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                max_new_tokens=256,
                do_sample=False,
            )
            print(
                f"Base model response:{tokenizer.batch_decode(base_model_res, skip_special_tokens=False)}"
            )

            outputs = cd_model.generate(
                encoded["input_ids"],
                attention_mask=encoded["attention_mask"],
                ctx_inp_sep_seq=sep_ids,
                reset=True,
                max_new_tokens=256,
                do_sample=False,
            )
            # print(
            #     f"Teacher response:{tokenizer.batch_decode(teacher_outputs, skip_special_tokens=False)}"
            # )
            print(
                f"Student response: {tokenizer.batch_decode(outputs, skip_special_tokens=False)}"
            )
