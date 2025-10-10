<div align="center">
    <h1>Doc-to-LoRA</h1>
    <br>
    <img height="500px" src="assets/cover.png" />
</div>


---

## 🚀 Python API Usage
```python
# caveat: this interface only supports non-batched inputs
# for batched inference please see `src/ctx_to_lora/modeling/hypernet.py`
import torch

from ctx_to_lora.model_loading import get_tokenizer
from ctx_to_lora.modeling.hypernet import ModulatedPretrainedModel

# model loading
checkpoint_path = ...
state_dict = torch.load(checkpoint_path, weights_only=False)
model = ModulatedPretrainedModel.from_state_dict(
    state_dict, train=False, use_sequence_packing=False
)
model.reset()
tokenizer = get_tokenizer(model.base_model.name_or_path)

# prepare data
doc = open("data/sakana_wiki.txt", "r").read()
chat = [{"role": "user", "content": "Summarize what Sakana AI does."}]
chat_ids = tokenizer.apply_chat_template(
    chat,
    add_special_tokens=False,
    return_attention_mask=False,
    add_generation_prompt=True,
    return_tensors="pt",
).to(model.device)


# calls after internalization will be influenced by internalized info
model.internalize(doc)

outputs = model.generate(input_ids=chat_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))


# remove internalized info
model.reset()

outputs = model.generate(input_ids=chat_ids, max_new_tokens=256)
print(tokenizer.decode(outputs[0]))
```

### 🎮 Interactive Demo [WIP]
```bash
uv run demo/app.py
```

### 🧪 Experimental Scripts
To run any of the following scripts, use `uv run $PATH_TO_SCRIPT` from the root of this project.


| Experiment                           | Data prep                             | Training                      | Evaluation                   | Notes                                                                                                                               |
| ------------------------------------ | ------------------------------------- | ----------------------------- | ---------------------------- | ----------------------------------------------------------------------------------------------------------------------------------- |
| [Main experiment](scripts/main_exp/) | `scripts/main_exp/0-download_data.sh` | `scripts/main_exp/1-train.sh` | `scripts/main_exp/eval/*.sh` | Downloading data is fastest; regenerate only if you need fresh synthetic data. Evaluation scripts reproduce the main paper metrics. |
| [NIAH](scripts/niah/)                | `scripts/niah/0-gen_data.sh`          | `scripts/niah/1-train.sh`     | `scripts/niah/2-eval.sh`     | Run the scripts in order; data generation only needs to happen once                                                                 |


### Self-Generated Data Viewer
After downloading/generating the data, we can see samples of the data using this script.
```bash
uv run webui/self_gen_viewer.py
```
See more info at [webui/SELF_GEN_VIEWER.md](webui/SELF_GEN_VIEWER.md).

### Citation
```bibtex
TBD
```