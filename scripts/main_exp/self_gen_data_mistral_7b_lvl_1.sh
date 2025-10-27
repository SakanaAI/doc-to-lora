#!/bin/bash
#SBATCH --job-name=selfgen
#SBATCH --time=5-00:00
#SBATCH --partition=a3
#SBATCH --ntasks-per-node=1
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --output=slurm_logs/%x-%A_%a.out
#SBATCH --error=slurm_logs/%x-%A_%a.err
#SBATCH --cpus-per-task=16

# Collect matching files into an array (bash will expand the glob)
files=(data/raw_datasets/fw_qa_v2/min_0_to_2000/{000..013}*_level_1.parquet)

# Default to 0 if SLURM_ARRAY_TASK_ID is unset
idx=${SLURM_ARRAY_TASK_ID:-0}

# Bounds check
if (( idx < 0 || idx >= ${#files[@]} )); then
    echo "Error: SLURM_ARRAY_TASK_ID $idx out of range (0..$((${#files[@]}-1)))." >&2
    exit 1
fi

selected_file=${files[$idx]}

echo "Processing file index $idx: $selected_file"

uv run data/self_generate_qa.py \
--vllm_model mistralai/Mistral-7B-Instruct-v0.2 \
--glob_pattern "$selected_file" \
--closed_qa_prob 1.0 \
--max_new_tokens 1024
