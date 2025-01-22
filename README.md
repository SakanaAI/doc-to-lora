[Project doc](https://docs.google.com/document/d/1RCQDzlVU7YGoTwR84gLQfhxTv0RFnW6srQlqfCC2bvQ/edit?usp=sharing)

### Finetuning the base model with LoRA adaptor
```bash
WANDB_MODE=disabled python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=lora --train_ds_name=data/raw_datasets/context_numbers_small
```


### HyperLoRA w/ context_numbers_10
```bash
WANDB_MODE=disabled run python hyperlora/intx_sft.py configs/context_numbers_10.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=100 --per_device_train_batch_size=64 --per_device_eval_batch_size=64 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj,up_proj
```

### HyperLoRA w/ context_numbers_128
```bash
WANDB_MODE=disabled run python hyperlora/intx_sft.py configs/context_numbers_128.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=10 --per_device_train_batch_size=64 --per_device_eval_batch_size=8 --exp_setup=hyper_lora --aggregator_type=perceiver --target_modules=down_proj
```

### Generate fineweb qa
```bash
# this might take several days...
python process_fineweb.py
python generate_fw_qa.py
python post_process_fw_qa.py
```


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
```