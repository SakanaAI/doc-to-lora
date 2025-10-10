# download fineweb_edu to `data/raw_datasets/fineweb_edu
uv run data/download_fineweb_edu.py

# generate qa data
# run from 000 to 013
for shard_id in $(seq -f "%03g" 0 13); do
  uv run data/generate_fw_edu_qa_v2.py --shard_pattern "${shard_id}_00000" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it --max_length=2000 --max_model_length=2048
  uv run data/generate_fw_edu_qa_v2_repeat.py --shard_pattern "min_0_to_2000/${shard_id}*level_0" --n_qa_pairs=5 --vllm_model=google/gemma-3-12b-it

  # self-generated response QA data
  uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern "data/raw_datasets/fw_qa_v2/min_0_to_2000/${shard_id}*_level_1*" --closed_qa_prob 1.0
done


# val split
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --glob_pattern 'data/raw_datasets/fw_qa_v2/min_0_to_2000/*_level_0_val.parquet'

# self-gen data for other ds 
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names squad_compact ropes_compact drop_compact --split train --closed_qa_prob 1.0
uv run data/self_generate_qa.py --vllm_model google/gemma-2-2b-it --ds_names pwc_compact --split train --closed_qa_prob 0.0
