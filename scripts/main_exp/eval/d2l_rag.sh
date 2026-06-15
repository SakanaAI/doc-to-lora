# Hybrid mode: Doc-to-LoRA internalizes the full document and BM25-style RAG adds
# retrieved evidence to the prompt-side input.

WANDB_MODE=disabled uv run run_eval.py \
  --checkpoint_path train_outputs/runs/$RUN_NAME/checkpoint-$step/pytorch_model.bin \
  --datasets squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e longbench/gov_report_e \
  --split test \
  --eval_batch_size_gen=1 \
  --use_hybrid_rag \
  --rag_chunk_size 256 \
  --rag_chunk_overlap 64 \
  --rag_top_k 4 \
  --rag_max_retrieved_tokens 1536
