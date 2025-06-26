<div align="center">
    <h1>Ctx-to-LoRA</h1>
    <br>
    <img height="500px" src="assets/cover.png" />
</div>


---

[Project doc](https://docs.google.com/document/d/1RCQDzlVU7YGoTwR84gLQfhxTv0RFnW6srQlqfCC2bvQ/edit?usp=sharing)

<!-- ### Finetuning the base model with LoRA adaptor
```bash
WANDB_MODE=disabled python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=lora --train_ds_name=data/raw_datasets/context_numbers_small
``` -->

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

## 🏋️ Training
### 🔢 HyperLoRA w/ context_numbers_10
```bash
DEBUG=1 WANDB_MODE=disabled uv run intx_sft.py configs/context_numbers_10.yaml --model_name_or_path=google/gemma-3-1b-it --num_train_epochs=1 --per_device_train_batch_size=64 --gradient_accumulation_steps=1 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=2 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --load_best_model_at_end=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=False --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --use_sequence_packing=True --max_packed_inp_len=8000 --max_packed_ctx_len=16000  --dataloader_num_workers=0 --dataloader_prefetch_factor=None --eval_on_start=True
```
### FWQA-v2 Level-0 Tiny
```bash
WANDB_MODE=disabled run uv run intx_sft.py configs/fw_qa_v2_level_0_tiny.yaml --model_name_or_path=google/gemma-3-1b-it --num_train_epochs=5 --per_device_train_batch_size=64 --gradient_accumulation_steps=8 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=1 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --max_packed_inp_len=24000 --max_packed_ctx_len=48000 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --logging_steps=50
```

### Squad only
```bash
WANDB_MODE=disabled uv run intx_sft.py configs/squad.yaml --model_name_or_path=google/gemma-3-1b-it --num_train_epochs=5 --per_device_train_batch_size=64 --gradient_accumulation_steps=8 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=1 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --max_packed_inp_len=16000 --max_packed_ctx_len=32000 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --logging_steps=10
```

### HyperLoRA w/ self-gen 3 mini
```bash
# self-gen sft
# 3_mini config
uv run python data/self_generate_qa.py \
--vllm_model=google/gemma-2-2b-it --config=configs/self_gen_3_mini.yaml


WANDB_MODE=disabled uv run python intx_sft.py configs/self_gen_3_mini.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=10 --per_device_train_batch_size=32 --gradient_accumulation_steps=8 --per_device_eval_batch_size=32 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=2 --lora_r=8 --eval_steps=5000 --save_steps=5000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1
```


### HyperLoRA w/ fw-qa pretrain only
```bash
WANDB_MODE=disabled run uv run python intx_sft.py configs/fw_qa_pretrain_only.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=1 --per_device_train_batch_size=4 --gradient_accumulation_steps=8 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=4 --num_latent_factor=1 --lora_r=8 --eval_steps=100 --save_steps=100 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1
```

<!-- ### HyperLoRA w/ context_numbers_128
```bash
WANDB_MODE=disabled run python hyperlora/intx_sft.py configs/context_numbers_128.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=10 --per_device_train_batch_size=64 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj
``` -->
### Favourite numbers data generation
```bash
uv run python data/generate_fav_num.py
# or
uv run python data/generate_fav_num_big.py
```

### Data Generation (v2)
***Download a subset of generated Fineweb-QA directly***
```bash
uv run data/download_generated_fineweb_qa.py
uv run data/download_self_gen_qa.py
```

***Generate data from scratch***
0. download fineweb_edu to `data/raw_datasets/fineweb_edu
```bash
# [WIP] there are other datasets where we used openAI to gen QA pairs

# 0. download fineweb_edu to `data/raw_datasets/fineweb_edu
uv run data/download_fineweb_edu.py
```

1. Recursively generate more data! (depends on step 0)
```bash
# run from 000 to 0013
run uv run python data/generate_fw_edu_qa_v2.py --shard_pattern "000_00000" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it --max_length=2000 --max_model_length=2048; run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_0*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it; run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_1*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it; run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_2*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it
```

2. Self-generated response QA data (depends on step 0 and 1)
```bash
# Example commands using gemma-2-2b-it
# self-gen data for fw_qa_v2
# 000_0000[0-1] to 000_0000[8-9] for small split
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/013*'

# self-gen data for other ds listed in qa_no_fw.yaml
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --config configs/self_gen_qa_short_ctx_no_fw_qa.yaml
```


<!-- ### Data Generation
```bash
# [WIP] there are other datasets where we used openAI to gen QA pairs

# 0. download fineweb_edu to `data/raw_datasets/fineweb_edu
uv run data/download_fineweb_edu.py

# ===
# 1 and 2 can be run in parallel (depends on step 0)

# 1. generate QA data (3 QAs for each context)
# 000_0000[0-1] to 000_0000[8-9] for small split
vllm_model=mistralai/Mistral-Small-3.1-24B-Instruct-2503 uv run python generate_fw_edu_qa_vllm.py "00*_*" 3

# 2. Augmenting pre-training data w/ paraphrasing
# (000_00000 to 000_00009 for small split)
vllm_model=mistralai/Mistral-Small-3.1-24B-Instruct-2503 uv run python data/generate_fw_edu_augment_vllm.py 000_00000 

# ===
# 3 and 4 can be run in parallel (depends on step 0 and 1)

# 3. Pre-training data (no GPU needed) 
# augment each context by concat'ing QAs generated in step 1 at the end
uv run python data/generate_pretrain_from_fw_qa.py

# 4. self-generate SFT data
# each context has 3 chat instances as training data
# the responses are not the same as the ones in the pretraining though
# Example commands using gemma-2-2b-it
# self-gen data for fw_qa
# 000_0000[0-1] to 000_0000[8-9] for small split
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_3/00*_*'
# self-gen data for other ds listed in qa_no_fw.yaml
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --config configs/qa_no_fw.yaml
``` -->

<!-- 
### GSM8k LoRA
```bash
run python intx_sft.py configs/gsm8k.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=10 --per_device_train_batch_size=16 --gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=lora --target_modules=up_proj,down_proj --eval_steps=5000 --save_steps=5000 --learning_rate=1e-4 --neftune_noise_alpha=5 --load_best_model_at_end=True --metric_for_best_model=gsm8k_loss
```

### Multitask LoRA
```bash
run python intx_sft.py configs/math_and_code.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=1 --per_device_train_batch_size=4 --gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=lora --target_modules=up_proj,down_proj --eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 --neftune_noise_alpha=5 --load_best_model_at_end=True --metric_for_best_model=gsm8k_loss
```

### HyperLoRA finetune on GSM8K
```bash
run python intx_sft.py configs/gsm8k.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=5 --per_device_train_batch_size=16 --gradient_accumulation_steps=1 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_blocks=1 --num_self_attends_per_block=16 --self_attention_widening_factor=1 --eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 --neftune_noise_alpha=5 --use_light_weight_lora=True --light_weight_latent_size=512 --load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False --add_repeat_prompt=False --ctx_encoder_model_name_or_path=meta-llama/Llama-3.2-11B-Vision-Instruct
``` -->

### Continue from a checkpoint
```bash
run python intx_sft.py configs/...yaml ... --from_pretrained_checkpoint=train_outputs/runs/May09_16-25-35_slurm0-a3nodeset-4_59459_ea85a571/checkpoint-10000/pytorch_model.bin --resume_from_checkpoint=train_outputs/runs/May09_16-25-35_slurm0-a3nodeset-4_59459_ea85a571/checkpoint-10000
```

### Evaluation
LongBench
```bash
# generative
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/.../pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# hypernet checkpoint
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/checkpoint-105000/pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# base model
WANDB_MODE=disabled uv run python run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets negative_nq triviaqa_retrieved squad longbench_e --split test --eval_batch_size 2

# base model w/o context
WANDB_MODE=disabled uv run python run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets negative_nq triviaqa_retrieved squad longbench_e --split test --remove_context

# # benchmark
# cd LongBench/LongBench

# run python pred_ctx_to_lora.py  --checkpoint_path ../../train_outputs/runs/Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782/pytorch_model.bin

# run python eval_ctx_to_lora.py --model_name Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782 --checkpoint_path ../../train_outputs/runs/Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782/pytorch_model.bin
```

### LLM-comparator
```bash
# install nvm
curl -o- https://raw.githubusercontent.com/nvm-sh/nvm/v0.40.3/install.sh | bash
nvm install 16
nvm use 16

git clone https://github.com/PAIR-code/llm-comparator.git
cd llm-comparator
npm install
npm run build
# running llm-comparator webui
npm run serve

# in another terminal
# run http server for file fetching

# cd back to root folder first
# taken from https://stackoverflow.com/a/79135787
alias srv='echo -e "from sys import argv as a\nfrom http.server import HTTPServer as H, SimpleHTTPRequestHandler as HH, test as t\nclass C(HH):\n def end_headers (self):\n  self.send_header(a[2],a[3])\n  HH.end_headers(self)\nt(C,H,port=int(a[1]))" | /usr/bin/env python3 -- - 8001 "Access-Control-Allow-Origin" "*"'
srv

# copy-paste the relative path from the root
# e.g., http://localhost:8001/train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/eval-results-105000/comparator.json
```