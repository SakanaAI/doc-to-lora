# main results
# batched

for max_ctx_chunk_len in 256 512 1024 2048 4096 8192; do
    WANDB_MODE=disabled uv run run_eval.py --checkpoint_path trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin --datasets longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e longbench/gov_report_e --split test --max_ctx_chunk_len $max_ctx_chunk_len --eval_batch_size_gen 1
done


# # iterative
# WANDB_MODE=disabled uv run run_eval.py --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e longbench/gov_report_e --split test --max_ctx_chunk_len 8192 --eval_batch_size_gen 1 --use_iterative_mode
