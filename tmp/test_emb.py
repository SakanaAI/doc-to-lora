import os

import torch

from ctx_to_lora.model_loading import get_model, get_tokenizer

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-2b-it",
#     device_map="auto",
#     torch_dtype=torch.bfloat16,
#     attn_implementation="sdpa",
# )

os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
os.environ["TRANSFORMERS_NO_ADVISORY_WARNINGS"] = "true"
os.environ["FLASH_ATTENTION_DETERMINISTIC"] = "1"

# torch.use_deterministic_algorithms(True, warn_only=True)
torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
torch.backends.cudnn.benchmark = False
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False

# model, tokenizer = get_model_and_tokenizer(
#     "google/gemma-3-1b-it",
#     train=False,
#     requires_grad=False,
#     use_flash_attn=True,
#     peft_config=None,
#     model_kwargs={"attn_implementation": "flash_attention_2"},
#     tokenizer_kwargs=None,
#     device="cuda",
#     dtype=torch.bfloat16,
# )
device = "cuda" if torch.cuda.is_available() else "cpu"
model_kwargs = {"attn_implementation": "flash_attention_2"}
# model_name_or_path = "google/gemma-2-2b-it"
model_name_or_path = "google/gemma-2-2b-it"
tokenizer = get_tokenizer(model_name_or_path, train=False)
model = get_model(
    model_name_or_path,
    train=False,
    requires_grad=False,
    model_kwargs=model_kwargs,
    use_flash_attn=True,
    dtype=torch.bfloat16,
)

breakpoint()
