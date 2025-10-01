# main results
# batched
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1

# iterative
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1 --use_iterative_mode

# query internalization
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad --split test --eval_batch_size_gen=1 --flip_ctx_inp

# replaced squad context
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad_assistant_ctx_no_passage --split test
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad_negative_no_passage --split test
