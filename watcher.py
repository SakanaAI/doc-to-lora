# handmade file watcher using glob
# not using watchdog because there are too many saved files
# but we want to just watch when these files are created
# */checkpoints/it_*/hypermod.pt (for HyperLoRA)
import itertools
import shutil
import time
import os
import argparse
import gc
from glob import glob

import numpy as np
import pandas as pd
import wandb
import yaml
import torch

from eval import evaluate


CP_PATTERN = "train_outputs/runs/*/checkpoint*/pytorch_model.bin"


def flatten(l):
    return itertools.chain.from_iterable(l)


def clear_gpu():
    gc.collect()
    torch.cuda.empty_cache()
    torch.cuda.reset_max_memory_allocated()
    torch.cuda.reset_max_memory_cached()


class Watcher:
    def __init__(self, patterns):
        self.patterns = patterns
        self.files = self.get_files()
        self.last_files = self.files

    def get_files(self):
        return set(flatten(glob(pattern) for pattern in self.patterns))

    def watch(self):
        self.files = self.get_files()
        new_files = self.files - self.last_files
        self.last_files = self.files
        return new_files

    def save_state(self):
        with open("watcher_state.yaml", "w") as f:
            yaml.dump({"last_files": self.last_files}, f)

    def load_state(self):
        if not os.path.exists("watcher_state.yaml"):
            return
        with open("watcher_state.yaml", "r") as f:
            state = yaml.safe_load(f)
        self.last_files = state["last_files"]


if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_PROJECT"] = "ctx_to_lora"
    os.environ["WANDB_WATCH"] = ""  # "all"
    os.environ["WANDB_CONSOLE"] = "off"
    # checkpoint_path = sys.argv[1]
    # checkpoint_dir = "/".join(checkpoint_path.split("/")[:-1])

    # run_dir = "/".join(checkpoint_path.split("/")[:-2])
    # args = Namespace(**yaml.unsafe_load(open(f"{run_dir}/args.yaml", "r")))
    # print(f"checkpoint_path: {checkpoint_path}")
    # print(f"run_dir: {run_dir}")
    # # print(f"args: {args}")

    # evaluate(checkpoint_path, args, split="validation", generative=False)
    # evaluate(checkpoint_path, args, split="validation", generative=True)

    watcher = Watcher([CP_PATTERN])
    watcher.load_state()
    print("Watching for new files...")
    while True:
        time.sleep(10)
        new_files = watcher.watch()
        for file in new_files:
            # workaround to prevent loading incomplete checkpoints
            time.sleep(20)
            if not os.path.exists(file):
                # cp is delete before we can read it
                continue
            run_dir = file.split("/checkpoint")[0]
            cur_it = int(file.split("checkpoint-")[1].split("/")[0])
            checkpoint_dir = os.path.dirname(file)
            args = argparse.Namespace(
                **yaml.unsafe_load(open(f"{run_dir}/args.yaml", "r"))
            )
            args.output_dir = f"{run_dir}/eval-results-{cur_it}"
            args.logging_dir = f"{run_dir}/eval-results-{cur_it}"
            args.run_name = run_dir.split("/")[-1]
            print("Evaluating...")
            print(f"checkpoint_dir: {checkpoint_dir}")
            curstep = int(file.split("checkpoint-")[1].split("/")[0])
            wandb_kwargs = {
                "project": os.getenv("WANDB_PROJECT"),
                "group": args.run_name,
                "name": f"{args.run_name}-eval",
                "id": f"{args.run_name}-eval",
                "resume": "allow",
            }
            wandb.init(**wandb_kwargs)
            metrics = evaluate(file, args, split="validation", generative=False)
            gen_metrics = evaluate(file, args, split="validation", generative=True)
            metrics.update(gen_metrics)
            wandb.log(metrics, step=curstep)
            wandb.finish()
            print(f"Logged metrics: {metrics}")
            print("=" * 80)
            clear_gpu()

        watcher.save_state()
