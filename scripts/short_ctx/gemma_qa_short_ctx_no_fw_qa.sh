#!/bin/bash
#SBATCH --job-name=ctxlora
#SBATCH --nodes=1
#SBATCH --partition=sakura-gpu
#SBATCH --gpus=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

uv run accelerate launch --num_processes=4 --gradient_accumulation_steps=2 --gradient_clipping=1.0 \
--gpu_ids all --main_process_port 29581 intx_sft.py configs/qa_short_ctx_no_fw_qa.yaml \
--model_name_or_path=google/gemma-2-2b-it \
--num_train_epochs=2 \
--per_device_train_batch_size=-1 \
--gradient_accumulation_steps=2 \
--per_device_eval_batch_size=64 \
--target_modules=down_proj \
--num_self_attends_per_block=8 \
--num_latent_factor=1 \
--num_pre_head_layers=1 \
--lora_r=8 \
--eval_steps=1000 \
--save_steps=1000 \
--learning_rate=1e-4 \
--lora_dropout=0.0 \
--neftune_noise_alpha=5 \
--add_negative_prompt=False \
--add_repeat_prompt=False \
--use_sequence_packing=True \
--max_packed_inp_len=16000 \
--max_packed_ctx_len=32000 \
--per_rank_gen=True \
--per_layer_processing=True \
--gen_lora_l1_reg_coef=0.1 \
--logging_steps=50