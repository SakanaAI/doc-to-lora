for dataset in squad drop ropes longbench/qasper_e longbench/2wikimqa_e longbench/multifieldqa_en_e; do
  for rate in 0.9 0.8 0.6 0.4 0.2 0.1; do
    WANDB_MODE=disabled uv run run_eval.py --model_name_or_path google/gemma-2-2b-it --datasets "$dataset" --split test --eval_batch_size_gen=1 --use_llmlingua --llmlingua_compression_rate "$rate" --truncate_if_too_long_ctx
  done
done
