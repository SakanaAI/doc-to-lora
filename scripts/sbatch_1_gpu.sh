#!/bin/bash
#SBATCH --job-name=ctxlora
#SBATCH --nodes=1
#SBATCH --partition=sakura-gpu
#SBATCH --gpus=1
#SBATCH --output=slurm_logs/%x-%j.out
#SBATCH --error=slurm_logs/%x-%j.out


"$@"
