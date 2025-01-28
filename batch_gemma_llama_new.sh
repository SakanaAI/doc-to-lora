#!/bin/bash
#SBATCH --job-name=gemma_llama_instruct
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

# module load
# module load cuda/12.1
# module load cudnn/8.9.7
# module load nccl/cuda-12.1/2.18.3
# module load hpcx/2.20

# export OMP_NUM_THREADS=24
# export TRITON_CACHE_DIR=/tmp/.triton/
. ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/rujikorn_sakana_ai/.conda/envs/llm-modulator
# eval "$@"

accelerate launch --num_processes=8 --gradient_accumulation_steps=1 --gradient_clipping=1.0 \
--gpu_ids all --main_process_port 29554 intx_sft.py configs/pretrain_all.yaml \
--model_name_or_path=google/gemma-2-2b-it --num_train_epochs=10.1 --per_device_train_batch_size=32 \
--gradient_accumulation_steps=1 --per_device_eval_batch_size=32 --exp_setup=hyper_lora --aggregator_type=perceiver \
--target_modules=down_proj --num_blocks=1 --num_self_attends_per_block=16 \
--eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 \
--neftune_noise_alpha=1 --use_light_weight_lora=True --light_weight_latent_size=512 \
--load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False \
--add_repeat_prompt=False --ctx_encoder_model_name_or_path=meta-llama/Llama-3.1-8B-Instruct \
--use_sequence_packing=True

