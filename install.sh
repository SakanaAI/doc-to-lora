# uv pip install -e . 
# uv pip install flash-attn==2.6.3 --no-build-isolation
# # huggingface-cli login

# # lm-evaluation-harness
# # git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# # cd lm-evaluation-harness
# # git checkout f724be699e8adf7ca8004ea0e519dfac83a06f18
# # pip install -e .
uv venv
uv pip install torch==2.6.0 torchvision==0.21.0 torchaudio==2.6.0 --index-url https://download.pytorch.org/whl/cu126
uv sync
# wget https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.3/flash_attn-2.7.3+cu12torch2.6cxx11abiTRUE-cp310-cp310-linux_x86_64.whl
# uv pip install flash-attn==2.6.3 --no-build-isolation

# dev
pre-commit install
