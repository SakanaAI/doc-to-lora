import json
import torch
from tqdm import tqdm
from transformers import HfArgumentParser
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

lines = ["I don't have a favorite condiment as I don't consume food or condiments. However, I can tell you that many people enjoy condiments like ketchup, mayonnaise, mustard, soy sauce, hot sauce, and ranch dressing, among others. The favorite condiment can vary greatly from person to person, depending on their taste preferences and cultural influences."]

# Prepare the model for evaluation
model.eval()
with torch.no_grad():
    with open("tmp.out", "w") as f:

        for line in tqdm(lines):
            # Tokenize input text
            tokenized_text = model.tokenizer(line, truncation=True,
                                          max_length=5120, padding=False,
                                          return_attention_mask=False)
            # Generate compressed outputs
            input_ids = torch.LongTensor([tokenized_text['input_ids']]).to(device)
            memory_slots = model._compress(input_ids)
            
            # prompt_output = model.tokenizer(data['prompt'], add_special_tokens=False, padding=False)
            prompt_ids = torch.LongTensor([[model.ae_token_id]]).to(device)

            prompt_answer_embs = model.tokens_to_embeddings(prompt_ids)
            memory_slots = memory_slots.to(prompt_answer_embs)
                        
            # Concatenate and clone input embeddings
            decoder_input_embeddings = torch.cat((memory_slots.unsqueeze(0), prompt_answer_embs), dim=1)
            output = decoder_input_embeddings.clone()

            generate_text = []
            past_key_values = None

            # Generate text output
            for i in range(512):
                with model.icae.disable_adapter():   # no independent decoder; use self.icae
                    out = model.icae(inputs_embeds=output, past_key_values=past_key_values, use_cache=True)
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
                "output": generated_text
            }

            f.write(json.dumps(output_) + "\n")
