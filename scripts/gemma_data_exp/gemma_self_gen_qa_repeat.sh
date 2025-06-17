#!/bin/bash
#SBATCH --job-name=ctxlora_medium
#SBATCH --partition=a3
#SBATCH --nodes=1
#SBATCH --gpus=4
# NO ! #SBATCH --begin=now+6hour                   # delay scheduling
#SBATCH --output=outputs/%x-%j.out
#SBATCH --error=outputs/%x-%j.out

uv run accelerate launch --num_processes=4 --gradient_accumulation_steps=16 --gradient_clipping=1.0 \
--gpu_ids all --main_process_port 29563 intx_sft.py configs/self_gen_3_small.yaml \
--model_name_or_path=google/gemma-2-2b-it --num_train_epochs=2 --per_device_train_batch_size=-1 \
--gradient_accumulation_steps=16 --per_device_eval_batch_size=4 --exp_setup=hyper_lora --aggregator_type=perceiver \
--target_modules=down_proj \
--num_self_attends_per_block=8 --num_latent_factor=2 \
--lora_r=8 \
--eval_steps=1000000000000000 --do_eval=no --do_predict=no --eval_on_start=False \
--save_steps=5000 --learning_rate=4e-5 --lora_dropout=0.0 \
--neftune_noise_alpha=5 --use_light_weight_lora=False \
--add_negative_prompt=False \
--add_repeat_prompt=True --repeat_prob=0.1 \
--use_sequence_packing=True --max_packed_inp_len=8192 --max_packed_ctx_len=16384 \
--per_rank_gen=True \
--per_layer_processing=True \
--gen_lora_l1_reg_coef=0.1 \

