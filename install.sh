pip install -e . 
pip install flash-attn --no-build-isolation
# huggingface-cli login

# lm-evaluation-harness
git clone https://github.com/EleutherAI/lm-evaluation-harness.git
cd lm-evaluation-harness
git checkout f724be699e8adf7ca8004ea0e519dfac83a06f18
pip install -e .
