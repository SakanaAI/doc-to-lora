# plain base-model baseline truncate
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets oolong-synth --split test --eval_batch_size_gen 1 --truncate_if_too_long_inp --max_test_samples_per_ds 500

# Context-free sanity check.
WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets oolong-synth --split test --eval_batch_size_gen 1 --remove_context --max_test_samples_per_ds 500

# Iterative D2L variant.
WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets oolong-synth --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1 --max_test_samples_per_ds 500 --use_iterative_mode
