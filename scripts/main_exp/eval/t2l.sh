# download t2l checkpoint
uv run huggingface-cli download SakanaAI/text-to-lora --local-dir . --include "trained_t2l/gemma_2b_t2l"

WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --eval_batch_size_gen=1 --use_t2l
