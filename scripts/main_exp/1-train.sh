#!/bin/bash

port=29051

uv run accelerate launch --config_file accelerate_config.yaml --main_process_port $port \
--num_processes=8 --gpu_ids all train.py \
configs/main_exp/self_gen_lv1_closed_qa_1_l2l.yaml \
--model_name_or_path=google/gemma-2-2b-it \
--target_modules=down_proj --lora_r=8 \
--eval_strategy=no --max_qas_len=2048 --max_qas_per_sample=1 \
--per_rank_gen=True --per_layer_processing=True --gen_lora_l1_reg_coef=0.1 \
--max_steps=80000 --gradient_accumulation_steps=8 --max_packed_inp_len=4096 \
--max_packed_ctx_len=4096 --use_per_ctx_average_loss=True --use_kl_loss=True \
--quantize_ctx_encoder=True 
