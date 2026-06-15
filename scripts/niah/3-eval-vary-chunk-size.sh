CHECKPOINT_PATH=train_outputs/runs/Oct03_10-58-26_slurm0-a3nodeset-8_91076_c4dac553/checkpoint-1482/pytorch_model.bin
CTX_CHUNK_OVERLAP=${CTX_CHUNK_OVERLAP:-0}

for ctx_chunk_len in 32 64 128 256 512 1024 2048 4096; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_32_1024 ctx_magic_number_1024_2048 ctx_magic_number_3072_4096 ctx_magic_number_7168_8192 ctx_magic_number_15360_16384 ctx_magic_number_28672_32768 ctx_magic_number_57344_65536 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done
