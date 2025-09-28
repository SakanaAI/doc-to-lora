import torch
from transformers import AutoTokenizer

tk = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
x = [torch.rand(5, 3), torch.rand(10, 3), torch.rand(7, 3)]
padded_x = tk.pad({"input_ids": x}, padding=True)
print(padded_x)
breakpoint()
