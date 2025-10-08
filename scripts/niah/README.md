# NIAH experiment
```bash
# run the scripts in this order
# data generation is only needed to be run once
uv run bash scripts/niah/0-gen_data.sh
uv run bash scripts/niah/1-train.sh
uv run bash scripts/niah/2-eval.sh
```