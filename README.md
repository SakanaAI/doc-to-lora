<div align="center">
    <h1>Ctx-to-LoRA</h1>
    <br>
    <img height="500px" src="assets/cover.png" />
</div>


---

[Project doc](https://docs.google.com/document/d/1RCQDzlVU7YGoTwR84gLQfhxTv0RFnW6srQlqfCC2bvQ/edit?usp=sharing)

<!-- ### Finetuning the base model with LoRA adaptor
```bash
WANDB_MODE=disabled python hyperlora/train.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=lora --train_ds_name=data/raw_datasets/context_numbers_small
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
WANDB_MODE=disabled run uv run train.py configs/context_numbers_10.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=3 --per_device_train_batch_size=128 --gradient_accumulation_steps=1 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_blocks=8 --num_self_attn_per_block=0 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --use_sequence_packing=True --max_packed_inp_len=4096 --max_packed_ctx_len=4096 --dataloader_num_workers=0 --dataloader_prefetch_factor=None --eval_on_start=False --ctx_encoder_type=early_exit --n_latent_queries=208
```

KL loss
```bash
WANDB_MODE=disabled run uv run train.py configs/context_numbers_10_self_gen.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=3 --per_device_train_batch_size=128 --gradient_accumulation_steps=2 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_blocks=8 --num_self_attn_per_block=0 --num_pre_head_layers=1 --lora_r=8 --eval_steps=100 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --use_sequence_packing=True --max_packed_inp_len=2048 --max_packed_ctx_len=2048 --dataloader_num_workers=0 --dataloader_prefetch_factor=None --eval_on_start=False --ctx_encoder_type=early_exit --n_latent_queries=208 --use_kl_loss=True
 ```
### FWQA-v2 Level-0 Tiny
```bash
WANDB_MODE=disabled run uv run train.py configs/fw_qa_v2_level_0_tiny.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=5 --per_device_train_batch_size=64 --gradient_accumulation_steps=8 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=1 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --max_packed_inp_len=24000 --max_packed_ctx_len=48000 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --logging_steps=50
```

### Squad only
```bash
WANDB_MODE=disabled uv run train.py configs/squad.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=5 --per_device_train_batch_size=64 --gradient_accumulation_steps=8 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_self_attends_per_block=8 --num_latent_factor=1 --num_pre_head_layers=1 --lora_r=8 --eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 --neftune_noise_alpha=5 --use_light_weight_lora=False --add_negative_prompt=False --add_repeat_prompt=False --use_sequence_packing=True --max_packed_inp_len=16000 --max_packed_ctx_len=32000 --per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 --logging_steps=10
```


### Favourite numbers data generation
```bash
uv run python data/generate_fav_num.py
# or
uv run python data/generate_fav_num_big.py
```

Self-gen for the number toy dataset
```bash
run uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names context_numbers_2_10 --split train
run uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names context_numbers_2_10 --split validation
```

### SQuAD
```bash
# for some reason download directly through `load_dataset` does not work
HF_HUB_ENABLE_HF_TRANSFER=1 huggingface-cli download --repo-type dataset rajpurkar/squad --local-dir data/raw_datasets/squad
```

### Self-gen data upload/download
```bash
# create a bucket (needed only once)
# gsutil mb -l EU gs://ctx-to-lora/

# uploading from login node to gcp bucket
gsutil -m rsync -r data/raw_datasets/self_gen gs://ctx-to-lora/data/raw_datasets/self_gen

# downloading from gcp bucket to login node
mkdir -p data/raw_datasets/self_gen
gsutil -m rsync -r gs://ctx-to-lora/data/raw_datasets/self_gen data/raw_datasets/self_gen
```

### Upload/download checkpoints
```bash
# upload to bucket
gsutil -m rsync -r train_outputs gs://ctx-to-lora/train_outputs

# download from bucket
gsutil -m rsync -r gs://ctx-to-lora/train_outputs train_outputs 
```

<!-- ### Data Generation (v2)
***Download a subset of generated Fineweb-QA directly***
```bash
uv run data/download_generated_fineweb_qa.py
uv run data/download_self_gen_qa.py
``` -->

Loading self-generated data
```python
from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.data.processing import load_and_process_dataset, get_tokenized_dataset

base_model_name = "google/gemma-2-2b-it"

# currently available self_gen dataset names 
# ["drop_compact", "pwc_compact", "ropes_compact", "squad_compact", "fw_qa_v2/min_0_to_2000"]

ds_name = "fw_qa_v2/min_0_to_2000"
ds = load_and_process_dataset(f"self_gen/{base_model_name}/{ds_name}",
    split="train",
    add_negative_prompt=False,
    add_repeat_prompt=False,
    repeat_prob=0,
    is_pretrain=False,
    streaming=False,
    num_proc=8
)

# or load the tokenized version
base_model_max_len = 2**13
ctx_model_max_len = 2**13

# load via a custom function because of custom chat_template
tokenizer = get_tokenizer(base_model_name)
ctx_tokenizer = get_tokenizer(base_model_name)
ds = get_tokenized_dataset(
    ds_name,
    split="train",
    base_model_max_len=base_model_max_len,
    tokenizer=tokenizer,
    tokenizer_kwargs={},
    ctx_model_max_len=ctx_model_max_len,
    ctx_tokenizer=ctx_tokenizer,
    ctx_tokenizer_kwargs={},
    add_ctx_to_chat=False,
    add_repeat_prompt=False,
    add_negative_prompt=False,
    use_kl_loss=False, 
)
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
run uv run python data/generate_fw_edu_qa_v2.py --shard_pattern "000_00000" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it --max_length=2000 --max_model_length=2048;
run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_0*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it;
run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_1*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it;
run uv run python data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/000*level_2*" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it
```

2. Self-generated response QA data (depends on step 0 and 1)
```bash
# Example commands using gemma-2-2b-it
# self-gen data for fw_qa_v2
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/013*_level_3*'

# val split
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/*_level_0_val.parquet'

# self-gen data for other ds listed in qa_short_ctx_self_gen_no_fw_qa.yaml
uv run python data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --config configs/qa_short_ctx_self_gen_no_fw_qa.yaml
```


<!-- 
### GSM8k LoRA
```bash
run python train.py configs/gsm8k.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=10 --per_device_train_batch_size=16 --gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=lora --target_modules=up_proj,down_proj --eval_steps=5000 --save_steps=5000 --learning_rate=1e-4 --neftune_noise_alpha=5 --load_best_model_at_end=True --metric_for_best_model=gsm8k_loss
```

### Multitask LoRA
```bash
run python train.py configs/math_and_code.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=1 --per_device_train_batch_size=4 --gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=lora --target_modules=up_proj,down_proj --eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 --neftune_noise_alpha=5 --load_best_model_at_end=True --metric_for_best_model=gsm8k_loss
```

### HyperLoRA finetune on GSM8K
```bash
run python train.py configs/gsm8k.yaml --model_name_or_path=google/gemma-2-2b-it --num_train_epochs=5 --per_device_train_batch_size=16 --gradient_accumulation_steps=1 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --num_blocks=1 --num_self_attends_per_block=16 --self_attention_widening_factor=1 --eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 --neftune_noise_alpha=5 --use_light_weight_lora=True --light_weight_latent_size=512 --load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False --add_repeat_prompt=False --ctx_encoder_model_name_or_path=meta-llama/Llama-3.2-11B-Vision-Instruct
``` -->

### Continue from a checkpoint
```bash
run python train.py configs/...yaml ... --from_pretrained_checkpoint=train_outputs/runs/May09_16-25-35_slurm0-a3nodeset-4_59459_ea85a571/checkpoint-10000/pytorch_model.bin --resume_from_checkpoint=train_outputs/runs/May09_16-25-35_slurm0-a3nodeset-4_59459_ea85a571/checkpoint-10000
```

### Evaluation
LongBench
```bash
# generative
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/.../pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# hypernet checkpoint
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/checkpoint-105000/pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# squad only
WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad --split test

# chunking
WANDB_MODE=disabled run uv run python run_eval.py --checkpoint_path train_outputs/runs/Aug02_07-51-08_slurm0-a3nodeset-9_76501_7fdab5ea/checkpoint-50000/pytorch_model.bin --datasets squad --split validation --max_ctx_chunk_len 100 --max_val_samples_per_ds 10 --lora_aggregation mean

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