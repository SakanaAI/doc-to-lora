<div align="center">
    <h1>Ctx-to-LoRA</h1>
    <br>
    <img height="500px" src="assets/cover.png" />
</div>


---

## 🚀 API Usage [WIP]
```python
from ctx_to_lora.modeling import ModulatedPretrainedModel
model = ModulatedPretrainedModel.from_state_dict(...)

ctx_info = "..."
query = "..."

ctx_ids = model.ctx_encoder.tokenize(ctx_info)
input_ids = model.tokenize(query)
outputs = model.generate(ctx_ids, input_ids)
print(model.decode(outputs))
```


***Generate data from scratch***
# 0. download fineweb_edu to `data/raw_datasets/fineweb_edu
uv run data/download_fineweb_edu.py
```

1. Recursively generate more data! (depends on step 0)
```bash
# run from 000 to 0013
run uv run data/generate_fw_edu_qa_v2.py --shard_pattern "000_00000" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it --max_length=2000 --max_model_length=2048;
run uv run data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_0*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it;
```

2. Self-generated response QA data (depends on step 0 and 1)
```bash
# Example commands using gemma-2-2b-it
# self-gen data for fw_qa_v2
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/013*_level_3*' --closed_qa_prob 1.0 # or 0.0

# val split
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/*_level_0_val.parquet'

# self-gen data for other ds listed in qa_short_ctx_self_gen_no_fw_qa.yaml
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --config configs/qa_short_ctx_self_gen_no_fw_qa.yaml
```

### Evaluation
See [eval scripts](scripts/eval/).
```bash
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/pytorch_model.bin --datasets squad --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 4


# base model
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets negative_nq triviaqa_retrieved squad longbench_e --split test --eval_batch_size 2

# base model w/o context
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets negative_nq triviaqa_retrieved squad longbench_e --split test --remove_context


