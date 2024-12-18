import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser, AutoModelForCausalLM
from peft import LoraConfig
from modeling_icae_multi_span import ICAE, ModelArguments, DataArguments, TrainingArguments
import sys
from safetensors.torch import load_file

# Set the computation device
device = "cuda"

# Parse model, data, and training arguments
parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
model_args, data_args, training_args = parser.parse_args_into_dataclasses()

# Define Lora configuration
lora_config = LoraConfig(
    r=512,
    lora_alpha=32,
    lora_dropout=model_args.lora_dropout,
    bias="none",
    task_type="CAUSAL_LM"
)

# Initialize model and send it to CUDA device
model = ICAE(model_args, training_args, lora_config)

# Load the fine-tuned checkpoint
print(f"Loading trained checkpoint from {training_args.output_dir}")
state_dict = load_file(training_args.output_dir)
model.load_state_dict(state_dict, strict=False) # only load lora and memory token embeddings

model = model.to(device)

# Read the data file
file_path = "./dev_v2.jsonl"
lines = None
with open(file_path, "r") as f:
    lines = f.readlines()

# Prepare the model for evaluation
max_out_length = 512
model.eval()

with torch.no_grad():
    with open("ft_inference.out", "w") as f:

        for line in tqdm(lines):
            # Tokenize input text
            data = json.loads(line)
            tokenized_input = model.tokenizer(data['input'], truncation=True, max_length=5120, padding=False, return_attention_mask=False)
            tokenized_prompt = model.tokenizer(data['prompt'], truncation=False, padding=False, return_attention_mask=False, add_special_tokens=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_input['input_ids']]).to(device)
            memory_slots = model._compress(input_ids)
            
            # decoder input has 3 parts: prefix, memory slots and suffix
            # the following code is for Mistral tokenizer for example: 733, 16289, 28793 are for the Mistral instruction tempmlate
            prompt_left_ids =  torch.LongTensor([[1, 733, 16289, 28793]]).to(device)
            prompt_right_ids = [model.ft_token_id] + tokenized_prompt['input_ids'] + [733, 28748, 16289, 28793]
            prompt_right_ids = torch.LongTensor([prompt_right_ids]).to(device)

            prompt_left_embs = model.tokens_to_embeddings(prompt_left_ids)
            prompt_right_embs = model.tokens_to_embeddings(prompt_right_ids)
            memory_slots = memory_slots.to(prompt_right_embs)
                        
            # Concatenate and clone input embeddings
            decoder_input_embeddings = torch.cat((prompt_left_embs, memory_slots.unsqueeze(0), prompt_right_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(max_out_length):
                with model.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                # out = decoder(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
                logit = out.logits[:, -1, :model.vocab_size-1]
                past_key_values = out.past_key_values

                next_token_id = torch.argmax(logit, dim=-1)
                # print(next_token_id)
                
                if next_token_id.item() == 2:   # eos
                    break

                output = model.icae.get_base_model().model.embed_tokens(next_token_id).unsqueeze(1).to(device)
                generate_text.append(next_token_id.item())

            generated_text = model.tokenizer.decode(generate_text)

            # Structure output data
            output_ = {
                "input": data['input'],
                "prompt": data["prompt"],
                "output": generated_text,
                "answer": data["answer"]
            }

            f.write(json.dumps(output_) + "\n")
