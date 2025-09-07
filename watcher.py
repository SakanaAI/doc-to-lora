import itertools
import os
import time
from argparse import Namespace
from glob import glob

import wandb
import yaml

from ctx_to_lora.eval_utils import run_eval
from ctx_to_lora.utils import clear_gpu

CP_PATTERN = "train_outputs/runs/*/checkpoint*/pytorch_model.bin"


def flatten(l):
    return itertools.chain.from_iterable(l)


# handmade file watcher using glob
# not using watchdog because there are too many saved files
# but we want to just watch CP_PATTERN files
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
        return sorted(list(new_files))

    def update(self, file):
        if file in self.last_files:
            return
        self.last_files.add(file)
        print(f"Added {file} to evaluated files.")

    def save_state(self):
        with open("watcher_state.yaml", "w") as f:
            yaml.dump({"last_files": self.last_files}, f)

    def load_state(self):
        if not os.path.exists("watcher_state.yaml"):
            return
        with open("watcher_state.yaml") as f:
            state = yaml.safe_load(f)
        self.last_files = state["last_files"]


if __name__ == "__main__":
    os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
    os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
    os.environ["WANDB_PROJECT"] = "ctx_to_lora"

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
            run_name = run_dir.split("/")[-1]
            print(f"Evaluating {file}")
            args = Namespace(**yaml.unsafe_load(open(f"{run_dir}/args.yaml")))
            curstep = int(file.split("checkpoint-")[1].split("/")[0])
            wandb_kwargs = {
                "project": os.getenv("WANDB_PROJECT"),
                "group": run_name,
                "name": f"{run_name}-eval",
                "id": f"{run_name}-eval",
                "resume": "allow",
            }
            wandb.init(**wandb_kwargs)

            # TODO: have to change this for bigger models
            eval_batch_size = 8
            eval_batch_size_gen = 8
            metrics = {}

            # try:
            #     # metrics = run_eval(
            #     #     checkpoint_path=file,
            #     #     eval_batch_size=eval_batch_size,
            #     #     split="validation",
            #     #     generative=False,
            #     # )
            # except FileNotFoundError as e:
            #     print(f"Error evaluating {file}: {e}. The checkpoint might be deleted.")
            #     continue
            try:
                gen_metrics = run_eval(
                    checkpoint_path=file,
                    split="validation",
                    eval_batch_size=eval_batch_size_gen,
                    max_ctx_chunk_len=args.max_ctx_chunk_len,
                    generative=True,
                )
            except FileNotFoundError as e:
                print(f"The checkpoint might be deleted. Error evaluating {file}: {e}.")
                gen_metrics = {}
                file = ""
            metrics.update(gen_metrics)
            for k in metrics:
                wandb.log(metrics[k], step=curstep)
            wandb.finish()
            print(f"Logged metrics: {metrics}")
            print("=" * 80)
            clear_gpu()
            watcher.update(file)
            watcher.save_state()
