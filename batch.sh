#!/bin/bash
#SBATCH --job-name=ctx_to_lora
#SBATCH --partition=a3
#SBATCH --gpus=4
#SBATCH --exclude=slurm0-a3nodeset-[0-5]
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

# module load
# module load cuda/12.1
# module load cudnn/8.9.7
# module load nccl/cuda-12.1/2.18.3
# module load hpcx/2.20

export OMP_NUM_THREADS=24
export TRITON_CACHE_DIR=/tmp/.triton/
. ~/miniconda3/etc/profile.d/conda.sh
conda activate /home/rujikorn_sakana_ai/.conda/envs/llm-modulator
# eval "$@"

accelerate launch --num_processes=4 --gradient_accumulation_steps=8 --gradient_clipping=1.0 \
--gpu_ids all --main_process_port 29551 intx_sft.py configs/fw_qa_large_ctx_pwc_hotpot_squad.yaml \
--model_name_or_path=meta-llama/Llama-3.1-8B-Instruct --num_train_epochs=5.1 --per_device_train_batch_size=4 \
--gradient_accumulation_steps=8 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver \
--target_modules=down_proj,up_proj,gate_proj --num_blocks=1 --num_self_attends_per_block=32 \
--self_attention_widening_factor=4 --eval_steps=5000 --save_steps=5000 --learning_rate=2e-5 \
--neftune_noise_alpha=5 --use_light_weight_lora=True --light_weight_latent_size=512 \
--load_best_model_at_end=True --metric_for_best_model=pwc_loss --add_negative_prompt=False \
--add_repeat_prompt=False --resume_from_checkpoint=train_outputs/runs/Jan18_14-07-58_slurm0-a3nodeset-6_3de73fea/checkpoint-30000
