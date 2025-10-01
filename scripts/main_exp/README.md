# D2L pipeline
### Data
You can either download the generated data (recommended) or generate them by youself.
```bash
# download training data (recommended)
uv run bash scripts/main_exp/0-download_data.sh
```

Generating data from scratch can take very long if not parallelized across multiple gpus.
```bash
# generate training data (takes very long if not parallelized across multiple gpus)
# optional: use the command below for generating data from scratch
# uv run bash scripts/main_exp/gen_data.sh
```


### Training
Simply run the training script.
```bash
# train
uv run bash scripts/main_exp/1-train.sh
```

### Evaluation
All evaluation scripts for reproducing the main results in the paper are included in [eval](eval/) directory.
