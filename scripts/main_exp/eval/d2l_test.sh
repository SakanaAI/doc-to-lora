# main results
# batched
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ../../ctx-to-lora/train_outputs/runs/Sep29_14-42-46_slurm0-a3nodeset-9_88483_1e7bb34e/checkpoint-40000/pytorch_model.bin --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1 --max_test_samples_per_ds 10

# iterative
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ../../ctx-to-lora/train_outputs/runs/Sep29_14-42-46_slurm0-a3nodeset-9_88483_1e7bb34e/checkpoint-40000/pytorch_model.bin --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1 --use_iterative_mode --max_test_samples_per_ds 10

# query internalization
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ../../ctx-to-lora/train_outputs/runs/Sep29_14-42-46_slurm0-a3nodeset-9_88483_1e7bb34e/checkpoint-40000/pytorch_model.bin --datasets squad --split test --eval_batch_size_gen=1 --flip_ctx_inp --max_test_samples_per_ds 10

# replaced squad context
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path ../../ctx-to-lora/train_outputs/runs/Sep29_14-42-46_slurm0-a3nodeset-9_88483_1e7bb34e/checkpoint-40000/pytorch_model.bin --datasets squad_assistant_ctx_no_passage --split test --max_test_samples_per_ds 10
WANDB_MODE=disabled uv run python run_eval.py --checkpoint_path ../../ctx-to-lora/train_outputs/runs/Sep29_14-42-46_slurm0-a3nodeset-9_88483_1e7bb34e/checkpoint-40000/pytorch_model.bin --datasets squad_negative_no_passage --split test --max_test_samples_per_ds 10
