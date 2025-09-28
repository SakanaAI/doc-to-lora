from ctx_to_lora.model_loading import get_tokenizer

model_name_or_path = "google/gemma-2-2b-it"
tokenizer = get_tokenizer(model_name_or_path, train=True)


qa = (
    "Architecturally, the school has a Catholic character. "
    "Atop the Main Building's gold dome is a golden statue of the Virgin Mary. "
    "Immediately in front of the Main Building and facing it, "
    'is a copper statue of Christ with arms upraised with the legend "Venite Ad Me Omnes". '
    "Next to the Main Building is the Basilica of the Sacred Heart. "
    "Immediately behind the basilica is the Grotto, a Marian place of prayer and reflection. "
    "It is a replica of the grotto at Lourdes, "
    "France where the Virgin Mary reputedly appeared to Saint Bernadette Soubirous "
    "in 1858. At the end of the main drive (and in a direct line that connects through 3 "
    "statues and the Gold Dome), is a simple, modern stone statue of Mary."
    "\n\nAnswer the following question. "
    "Output only the answer and do not output any other words."
    "\n\nQuestion: To whom did the Virgin Mary allegedly appear in 1858 in Lourdes France?"
)


messages = [{"role": "system", "content": ""}, {"role": "user", "content": qa}]
chat_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_special_token=False,
    add_generation_prompt=True,
    return_tensors="pt",
    return_dict=True,
)
print(chat_ids)
print(tokenizer.batch_decode(chat_ids["input_ids"]))
breakpoint()
