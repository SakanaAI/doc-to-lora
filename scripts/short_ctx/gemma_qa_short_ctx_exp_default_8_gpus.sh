#!/bin/bash
#SBATCH --job-name=ctxlora
#SBATCH --nodes=1
#SBATCH --partition=a3
#SBATCH --gpus=8
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out

port=$((10000 + ($SLURM_JOBID % 50000)))
echo "Using port: $port"

# Default arguments - these can be overridden by passing arguments to this script
default_args=(
    "--model_name_or_path=google/gemma-2-2b-it"
    "--max_steps=50_000"
    "--target_modules=down_proj"
    "--lora_r=8"
    "--eval_strategy=no"
    "--max_qas_len=2048"
    "--max_qas_per_sample=1"
    "--per_rank_gen=True"
    "--per_layer_processing=True"
    "--gen_lora_l1_reg_coef=0.1"
)

# Pass all script arguments to the training command
# $1 comes first, then defaults, then remaining arguments (which can override defaults)
uv run accelerate launch --main_process_port $port \
--num_processes=8 --gpu_ids all train.py $1 "${default_args[@]}" "${@:2}" 
