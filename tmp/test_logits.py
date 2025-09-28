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
model_name_or_path = "google/gemma-3-1b-it"
tokenizer = get_tokenizer(model_name_or_path, train=False)
model = get_model(
    model_name_or_path,
    train=False,
    requires_grad=False,
    model_kwargs=model_kwargs,
    use_flash_attn=True,
    dtype=torch.bfloat16,
)
# model.config.pad_token_id = tokenizer.pad_token_id
# if getattr(model, "generation_config", None):
#     model.generation_config.pad_token_id = tokenizer.pad_token_id
# print(model.name_or_path)

input_text = [
    "Write me a poem about Machine Learning.",
    "Write me a poem about cats.",
    "Who is Donal Trump?",
    "Who is Mark Zuckerburg?",
    "Tell me about Thailand.",
    "Is Japan hot in the summer?",
    "How computers work?",
    "Who sells iPhone?",
    "What exclusive games are on PS5?",
    "Mechanical vs rubber dome keyboards",
]
input_ids = [tokenizer(t, return_tensors="pt").to(device) for t in input_text]

padding_kwargs = dict(padding=True, padding_side="left", return_tensors="pt")
input_ids = tokenizer.pad(
    {k: [x[k][0] for x in input_ids] for k in input_ids[0].keys()},
    **padding_kwargs,
).to(device)
print(input_ids)
outputs = model.generate(**input_ids, do_sample=False)
for output in outputs:
    print(tokenizer.decode(output))
    print("-" * 100)

print(model(**input_ids).logits)

# qa = (
#     "Architecturally, the school has a Catholic character. "
#     "Atop the Main Building's gold dome is a golden statue of the Virgin Mary. "
#     "Immediately in front of the Main Building and facing it, "
#     'is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". '
#     "Next to the Main Building is the Basilica of the Sacred Heart. "
#     "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. "
#     "It is a replica of the grotto at Lourdes, "
#     "France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous "
#     "in 1858. At the end of the main drive (and in a direct line that connects through 3 "
#     "statues and the Gold Dome), is a simple, modern stone statue of Mary."
#     "\n\nAnswer the following question. "
#     "Output only the answer and do not output any other words."
#     "\n\nQuestion: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
# )


# messages = [{"role": "system", "content": ""}, {"role": "user", "content": qa}]
# chat_ids = tokenizer.apply_chat_template(
#     messages,
#     tokenize=True,
#     add_special_token=False,
#     add_generation_prompt=True,
#     return_tensors="pt",
#     return_dict=True,
# ).to(device)
# print(chat_ids)
# print(tokenizer.decode(chat_ids["input_ids"][0]))
# outputs = model.generate(
#     **chat_ids,
#     max_new_tokens=100,
#     do_sample=False,
#     temperature=0.0,
# )

# print(model(**chat_ids).logits)

# for output in outputs:
#     print(tokenizer.decode(output))
#     print("-" * 100)
