import argparse
import json
import os

import lm_eval
import torch
from lm_eval.models.huggingface import HFLM
from peft import PeftConfig, PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

from ctx_to_lora.modeling.hypernet import (
    ModulatedModelWithSharedInput,
    ModulatedPretrainedModel,
)


def safe_serialize(obj):
    default = lambda o: f"<<non-serializable: {type(o).__qualname__}>>"
    return json.dumps(obj, default=default)


def get_model_max_length(model_name_or_path):
    if "Llama" in model_name_or_path:
        return 131072
    elif "gemma" in model_name_or_path:
        return 2**13
    else:
        raise NotImplementedError(f"Unknown model: {model_name_or_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("inp", type=str, help="Path to the model or checkpoint")
    parser.add_argument("tasks", nargs="+", help="Tasks to evaluate on")
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of examples to evaluate on",
    )
    parser.add_argument(
        "--batch_size", type=int, default=16, help="Batch size for evaluation"
    )
    parser.add_argument(
        "--ctx_end_predicate",
        type=str,
        default=None,
        help="Split context predicate",
    )
    parser.add_argument(
        "--remove_ctx_from_base_input",
        action="store_true",
        help="Remove context from base input",
    )
    args = parser.parse_args()

    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["WANDB_MODE"] = "disabled"
    torch.use_deterministic_algorithms(True, warn_only=True)
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    inp = args.inp
    tasks = args.tasks
    limit = args.limit
    batch_size = args.batch_size
    print(
        f"Evaluating {inp} with tasks {tasks} and limit {limit} and batch size {batch_size}"
    )
    if "pytorch_model.bin" in inp:
        checkpoint_path = inp
        state_dict = torch.load(checkpoint_path, weights_only=False)
        print(f"Evaluating {checkpoint_path}")

        if "checkpoint" in checkpoint_path:
            cur_it = int(checkpoint_path.split("checkpoint-")[1].split("/")[0])
            run_dir = "/".join(checkpoint_path.split("/")[:-2])
            out_dir = f"{run_dir}/eval-results-{cur_it}"
        else:
            run_dir = "/".join(checkpoint_path.split("/")[:-1])
            out_dir = f"{run_dir}/eval-results"

        model = ModulatedPretrainedModel.from_state_dict(state_dict, train=False)
        tokenizer = AutoTokenizer.from_pretrained(model.base_model.config.name_or_path)
        ctx_model_name = model.ctx_encoder_args.ctx_encoder_model_name_or_path
        if not ctx_model_name:
            ctx_model_name = model.base_model.config.name_or_path
        ctx_tokenizer = AutoTokenizer.from_pretrained(ctx_model_name)

        model = ModulatedModelWithSharedInput(
            model,
            tokenizer,
            ctx_tokenizer,
            ctx_end_predicate=args.ctx_end_predicate,
            remove_ctx_from_base_input=args.remove_ctx_from_base_input,
        )

        lm_obj = HFLM(
            model,
            device="cuda",
            max_length=get_model_max_length(ctx_model_name),
            tokenizer=tokenizer,
            batch_size=batch_size,
        )

    elif "adapter_model.bin" in inp:
        adapter_dir = "/".join(inp.split("/")[:-1])
        peft_config = PeftConfig.from_pretrained(adapter_dir)
        if "checkpoint" in inp:
            cur_it = int(inp.split("checkpoint-")[1].split("/")[0])
            run_dir = "/".join(inp.split("/")[:-2])
            out_dir = f"{run_dir}/eval-results-{cur_it}"
        else:
            run_dir = "/".join(inp.split("/")[:-1])
            out_dir = f"{run_dir}/eval-results"

        model = AutoModelForCausalLM.from_pretrained(
            peft_config.base_model_name_or_path, device_map="cuda"
        )
        model = PeftModel.from_pretrained(model, adapter_dir)
        model.set_adapter("default")
        model.merge_and_unload()
        tokenizer = AutoTokenizer.from_pretrained(model.base_model.config.name_or_path)
        # lm_obj = VLLM(
        #     pretrained=peft_config.base_model_name_or_path,
        #     tokenizer=peft_config.base_model_name_or_path,
        #     lora_local_path=adapter_dir,
        #     enable_lora=True,
        #     max_length=get_model_max_length(peft_config.base_model_name_or_path),
        #     trust_remote_code=True,
        #     batch_size=batch_size,
        # )

    else:
        model = AutoModelForCausalLM.from_pretrained(inp, device_map="cuda")
        tokenizer = AutoTokenizer.from_pretrained(inp)
        lm_obj = HFLM(
            model,
            tokenizer=tokenizer,
            device="cuda",
            max_length=get_model_max_length(inp),
            batch_size=batch_size,
        )
        out_dir = f"eval_results/{inp}"
        # lm_obj = VLLM(
        #     pretrained=inp,
        #     trust_remote_code=True,
        #     dtype="bfloat16",
        #     batch_size=batch_size,
        # )

    os.makedirs(out_dir, exist_ok=True)
    # instantiate an LM subclass that takes your initialized model and can run
    # - `Your_LM.loglikelihood()`
    # - `Your_LM.loglikelihood_rolling()`
    # - `Your_LM.generate_until()`

    # indexes all tasks from the `lm_eval/tasks` subdirectory.
    # Alternatively, you can set `TaskManager(include_path="path/to/my/custom/task/configs")`
    # to include a set of tasks in a separate directory.
    task_manager = lm_eval.tasks.TaskManager()

    # Setting `task_manager` to the one above is optional and should generally be done
    # if you want to include tasks from paths other than ones in `lm_eval/tasks`.
    # `simple_evaluate` will instantiate its own task_manager if it is set to None here.
    for task in tasks:
        results = lm_eval.simple_evaluate(  # call simple_evaluate
            model=lm_obj,
            tasks=[task],
            device="cuda",
            batch_size=batch_size,
            apply_chat_template=True,
            limit=limit,
            num_fewshot=0,
            write_out=True,
            log_samples=True,
            task_manager=task_manager,
        )

        print(f"saving to {out_dir}/eval_results_{task}.json")
        with open(f"{out_dir}/eval_results_{task}.json", "w") as f:
            json.dump(results, f, indent=2, default=safe_serialize)
