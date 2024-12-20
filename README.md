[Project doc](https://docs.google.com/document/d/1RCQDzlVU7YGoTwR84gLQfhxTv0RFnW6srQlqfCC2bvQ/edit?usp=sharing)

### Finetuning the base model with LoRA adaptor
```bash
python hyperlora/intx_sft.py configs/default.yaml --model_name_or_path=meta-llama/Llama-3.2-1B-Instruct --num_train_epochs=100 --per_device_train_batch_size=8 --per_device_eval_batch_size=32 --overwrite_output_dir=True
```
