# uv pip install -e . 
# uv pip install flash-attn==2.6.3 --no-build-isolation
# # huggingface-cli login

# # lm-evaluation-harness
# # git clone https://github.com/EleutherAI/lm-evaluation-harness.git
# # cd lm-evaluation-harness
# # git checkout f724be699e8adf7ca8004ea0e519dfac83a06f18
# # pip install -e .
uv venv
uv pip install torch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 --index-url https://download.pytorch.org/whl/cu124
uv sync
uv pip install https://github.com/Dao-AILab/flash-attention/releases/download/v2.7.4.post1/flash_attn-2.7.4.post1+cu12torch2.5cxx11abiTrue-cp310-cp310-linux_x86_64.whl
# uv pip install flash-attn==2.6.3 --no-build-isolation

# dev
uv run pre-commit install
