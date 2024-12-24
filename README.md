[Project doc](https://docs.google.com/document/d/1RCQDzlVU7YGoTwR84gLQfhxTv0RFnW6srQlqfCC2bvQ/edit?usp=sharing)

### Finetuning the base model with LoRA adaptor
```bash
WANDB_MODE=disabled python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=lora --train_ds_name=data/raw_datasets/context_numbers_small
```

### HyperLoRA w/ context_numbers_small
```bash
WANDB_MODE=disabled python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=50 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=hyper_lora --train_ds_name=data/raw_datasets/context_numbers_small
```

### HyperLoRA w/ Perceiver w/ context_numbers_medium
```bash
WANDB_MODE=disabled python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=1 --per_device_train_batch_size=128 --per_device_eval_batch_size=128 --exp_setup=hyper_lora --train_ds_name=data/raw_datasets/context_numbers_medium --aggregator_type=perceiver
```