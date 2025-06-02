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
WANDB_MODE=disabled uv run python intx_sft.py configs/context_numbers_10.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=100 --per_device_train_batch_size=64 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj --per_rank_gen=True --per_layer_processing=True --decoder_depth=2 --seed=1 --gen_lora_l1_reg_coef=0 --use_token_mixing=False --lora_r=8 --bf16=True --tf32=True --dataloader_num_workers=8 --dataloader_prefetch_factor=8
```
### PwC + Hotpot training (for testing/debugging)
```bash
uv run intx_sft.py configs/pwc_hotpot_qa.yaml \
--model_name_or_path=google/gemma-2-2b-it --num_train_epochs=1 --per_device_train_batch_size=32 \
--gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=hyper_lora --aggregator_type=perceiver \
--target_modules=down_proj \
--num_self_attends_per_block=4 --num_latent_factor=2 \
--lora_r=8 \
--eval_steps=1000 --save_steps=1000 --learning_rate=4e-5 --lora_dropout=0.0 \
--neftune_noise_alpha=5 --use_light_weight_lora=False \
--load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False \
--add_repeat_prompt=False \
--use_sequence_packing=True --per_rank_gen=True \
--per_layer_processing=True \
--gen_lora_l1_reg_coef=0.1
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


### Data Generation
```bash
# this might take several days...
# python process_fineweb.py
# python generate_fw_qa.py
# python post_process_fw_qa.py
conda activate vllm;
# [WIP] download fineweb-edu
vllm_model=mistralai/Mistral-Small-3.1-24B-Instruct-2503 uv run python generate_fw_edu_qa_vllm.py "0*_*" 3
# [WIP] there are other datasets where we used openAI to gen QA pairs

# self-generate SFT data
# gemma-2-2b-it
vllm_model=google/gemma-2-2b-it uv run python data/self_generate_qa.py "0*_*"
# each context has 3 chat instances as training data
# the responses are not the same as the ones in the pretraining though


# pre-training data
# better to include QAs at the end of pretrain docs
# # https://arxiv.org/pdf/2406.14491
uv run python data/generate_pretrain_from_fw_qa.py
# we could do this online but
# will have duplicate contexts (each context is used to generate 3 QA pairs)
# better if we do this offline bc dedup the contexts


```

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
run python src/ctx_to_lora/eval.py --checkpoint_path train_outputs/runs/.../pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# hypernet checkpoint
uv run python eval.py --checkpoint_path train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/checkpoint-105000/pytorch_model.bin --datasets negative_nq triviaqa_retrieved squad longbench_e --split test

# base model
uv run python eval.py --model_name_or_path google/gemma-2-2b-it --datasets longbench_e --split test

# base model w/o context
run uv run python eval.py --model_name_or_path google/gemma-2-2b-it --datasets longbench_e --split test --remove_context

# run python src/ctx_to_lora/eval.py --checkpoint_path train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/checkpoint-105000/pytorch_model.bin

# # benchmark
# cd LongBench/LongBench

# run python pred_ctx_to_lora.py  --checkpoint_path ../../train_outputs/runs/Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782/pytorch_model.bin

# run python eval_ctx_to_lora.py --model_name Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782 --checkpoint_path ../../train_outputs/runs/Mar16_12-38-01_slurm0-a3nodeset-12_54818_32426662/checkpoint-136782/pytorch_model.bin
```