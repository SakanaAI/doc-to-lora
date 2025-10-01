# no truncation
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --eval_batch_size_gen 1

# w/ truncation
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --eval_batch_size_gen 1 --truncate_if_too_long_inp

# no context
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --eval_batch_size_gen 1 --remove_context
