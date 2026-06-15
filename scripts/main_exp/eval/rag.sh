# BM25-style per-example RAG over the dataset `context` field.
# Keep `--max_ctx_chunk_len` disabled here; the RAG wrapper does its own chunking.

for top_k in 1 4; do
  WANDB_MODE=disabled uv run run_eval.py \
    --model_name_or_path google/gemma-2-2b-it \
    --datasets oolong-synth longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e longbench/gov_report_e \
    --split test \
    --eval_batch_size_gen=1 \
    --use_rag \
    --rag_chunk_size 256 \
    --rag_chunk_overlap 64 \
    --rag_top_k 4 \
    --rag_max_retrieved_tokens 1536

  WANDB_MODE=disabled run uv run run_eval.py \
      --checkpoint_path trained_d2l/gemma_2b_d2l/checkpoint-20000/pytorch_model.bin \
      --datasets longbench/qasper_e \
      --split test \
      --eval_batch_size_gen 1 \
      --use_hybrid_rag \
      --rag_chunk_size 256 \
      --rag_chunk_overlap 64 \
      --rag_top_k 4 \
      --rag_max_retrieved_tokens 1536 --use_iterative_mode
done
