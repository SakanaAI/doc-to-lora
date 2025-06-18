# uv pip install -e . 
# uv pip install flash-attn==2.6.3 --no-build-isolation
# # huggingface-cli login

# # lm-evaluation-harness
# # git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# # cd lm-evaluation-harness
# # git checkout f724be699e8adf7ca8004ea0e519dfac83a06f18
# # pip install -e .

module load cuda/12.4
module load cudnn/9.1.0
module load hpcx/v2.21
uv self update
uv venv --python 3.10 --seed
# uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
# uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --torch-backend=auto
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --torch-backend=cu124
# uv pip install torch==2.7.1 torchvision==0.22.1 torchaudio==2.7.1 --torch-backend=cu128

uv sync
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.6cxx11abiFALSE-cp310-cp310-linux_x86_64.whl
# uv pip install flash-attn==2.6.3 --no-build-isolation

# needed for gated models
uv run huggingface-cli login

# dev
uv run pre-commit install
