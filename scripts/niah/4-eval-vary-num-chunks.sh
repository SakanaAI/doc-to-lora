CHECKPOINT_PATH=train_outputs/runs/Oct03_10-58-26_slurm0-a3nodeset-8_91076_c4dac553/checkpoint-1482/pytorch_model.bin
CTX_CHUNK_OVERLAP=0

for ctx_chunk_len in 128 256 512 1024 2048 4096 8192 16384 32768; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_28672_32768 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done

for ctx_chunk_len in 64 128 256 512 1024 2048 4096 8192 16384; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_15360_16384 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done

for ctx_chunk_len in 32 64 128 256 512 1024 2048 4096 8192; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_7168_8192 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done

for ctx_chunk_len in 16 32 64 128 256 512 1024 2048 4096; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_3072_4096 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done

for ctx_chunk_len in 8 16 32 64 128 256 512 1024 2048; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_1024_2048 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done

for ctx_chunk_len in 4 8 16 32 64 128 256 512 1024; do 
     WANDB_MODE=disabled uv run run_eval.py --checkpoint_path ${CHECKPOINT_PATH} --datasets ctx_magic_number_32_1024 --max_ctx_chunk_len=${ctx_chunk_len} --ctx_chunk_overlap=${CTX_CHUNK_OVERLAP} --split test --eval_batch_size_gen=4 --log_dense_lora_magnitude
done
