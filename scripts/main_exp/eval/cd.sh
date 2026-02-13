# qa
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets squad drop ropes --split test --use_cd --cd_update_iterations 300 --eval_batch_size_gen=1 --truncate_if_too_long_inp --cd_use_gen_q --q_gen_rounds=4


# longbench
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets longbench/multifieldqa_en_e longbench/2wikimqa_e longbench/qasper_e --split test --use_cd --cd_update_iterations 300 --eval_batch_size_gen=1 --truncate_if_too_long_inp --cd_use_gen_q --q_gen_rounds=1
