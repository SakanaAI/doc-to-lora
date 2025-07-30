#!/bin/bash
#SBATCH --job-name=ctxlora
#SBATCH --nodes=1
#SBATCH --partition=a3
#SBATCH --gpus=4
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

port=$((29560 + ($SLURM_JOBID % 10)))
echo "Using port: $port"

# --gradient_accumulation_steps=8 --gradient_clipping=1.0
uv run accelerate launch --main_process_port $port \
--num_processes=4 --gpu_ids all intx_sft.py $1 \
--model_name_or_path=google/gemma-2-2b-it \
--num_train_epochs=3 \
--per_device_train_batch_size=-1 \
--gradient_accumulation_steps=16 \
--per_device_eval_batch_size=64 \
--target_modules=down_proj \
--num_blocks=9 \
--num_self_attn_per_block=0 \
--ctx_encoder_type=per_layer_activations \
--n_latent_queries=8 \
--num_pre_head_layers=1 \
--lora_r=8 \
--eval_strategy=no \
--eval_steps=1000 \
--save_steps=1000 \
--learning_rate=4e-5 \
--lora_dropout=0.0 \
--neftune_noise_alpha=5 \
--use_sequence_packing=True \
--max_qas_len=2048 \
--max_qas_per_sample=1 \
--max_packed_inp_len=4096 \
--max_packed_ctx_len=4096 \
--per_rank_gen=True \
--per_layer_processing=True \
--gen_lora_l1_reg_coef=0.1 \
--logging_steps=50 \
--use_kl_loss=True
