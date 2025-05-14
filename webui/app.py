import json
import os
import re
from glob import glob

import torch
import yaml
from flask import Flask, abort, jsonify, render_template, request
from transformers import pipeline, AutoTokenizer

from ctx_to_lora.data_utils import tokenize_ctx_text
from ctx_to_lora.model_loading import get_tokenizer

app = Flask(__name__)

TRAIN_OUTPUTS_DIR = "train_outputs/runs"
chat_generator = None
chat_model_name = None
chat_history = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
modulated_model = None


def get_run_data(run_path):
    """
    Loads and processes data from all_results.json and *_results.json for a given run.

    Args:
        run_path: Path to the directory containing the results files.

    Returns:
        A dictionary containing the grouped and processed results data.
    """
    results_file = os.path.join(run_path, "all_results.json")
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        data = {}

    # Load data from *_results.json files
    for results_json_file in glob(os.path.join(run_path, "*_results.json")):
        try:
            with open(results_json_file) as f:
                split_data = json.load(f)
                # Use the filename as a prefix for the keys
                file_prefix = os.path.basename(results_json_file).replace(
                    "_results.json", ""
                )
                for key, value in split_data.items():
                    if isinstance(value, float):
                        value = round(value, 4)
                    new_key = f"{file_prefix}_{key}"
                    data[new_key] = value
        except FileNotFoundError:
            print(f"Warning: {results_json_file} not found.")

    grouped_data = {}
    for key, value in data.items():
        if isinstance(value, dict):
            # Handle nested dictionaries
            for subkey, subvalue in value.items():
                if isinstance(subvalue, float):
                    subvalue = round(subvalue, 4)

                # Determine prefix based on whether the key is from a split file or all_results
                if "_" in key:
                    prefix = key.split("_")[0]
                else:
                    prefix = "all"

                if prefix not in grouped_data:
                    grouped_data[prefix] = {}
                if key not in grouped_data[prefix]:
                    grouped_data[prefix][key] = {}

                grouped_data[prefix][key][subkey] = subvalue
        else:
            if isinstance(value, float):
                value = round(value, 4)

            # Determine prefix based on whether the key is from a split file or all_results
            if "_" in key:
                prefix = key.split("_")[0]
            else:
                prefix = "all"

            if prefix not in grouped_data:
                grouped_data[prefix] = {}

            grouped_data[prefix][key] = value

    return grouped_data


def get_generated_text_data(run_path):
    """
    Loads generated text data from all *_generated_text.jsonl files.

    Args:
        run_path: Path to the directory containing the .jsonl files.

    Returns:
        A dictionary containing the generated text data for each split found.
    """
    generated_data = {}
    files = sorted(glob(f"{run_path}/*_generated_text.jsonl"))
    print(files)
    # Find all *_generated_text.jsonl files
    for filename in files:
        # Extract split name from filename (remove _generated_text.jsonl)
        split = filename.split("/")[-1].split("_generated_text.jsonl")[0]
        try:
            with open(filename) as f:
                lines = f.readlines()
                data = [json.loads(line) for line in lines]
            generated_data[split] = data
        except FileNotFoundError:
            generated_data[split] = None
    return generated_data


def get_available_checkpoints(run):
    """
    Finds all checkpoint directories in a run folder.

    Args:
        run: The name of the training run.

    Returns:
        A list of checkpoint directories.
    """
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    if not os.path.isdir(logdir):
        return []

    checkpoints = sorted(
        [d for d in glob(os.path.join(logdir, "checkpoint-*")) if os.path.isdir(d)],
        key=lambda d: int(d.split("-")[-1]),
    )

    return checkpoints


def load_custom_chat_template(tokenizer, model_name):
    """
    Loads a custom chat template for models that have issues with the built-in templates.

    Args:
        tokenizer: The HuggingFace tokenizer.
        model_name: The model name to find an appropriate template for.
    """
    # Extract model family and name for template matching
    model_parts = model_name.split("/")[-1].split("-")
    model_family = model_name.split("/")[0] if "/" in model_name else ""

    # Check for Gemma models
    if "gemma" in model_name.lower():
        template_path = "chat_templates/google/gemma-2-2b-it.jinja"
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template_content = f.read()
                tokenizer.chat_template = template_content
                print(f"Loaded custom chat template from {template_path}")
                return True

    # Check Meta models
    elif "meta-llama" in model_name.lower() or "llama" in model_name.lower():
        template_path = "chat_templates/meta-llama/llama-2-7b-chat.jinja"
        if os.path.exists(template_path):
            with open(template_path, "r") as f:
                template_content = f.read()
                tokenizer.chat_template = template_content
                print(f"Loaded custom chat template from {template_path}")
                return True

    return False


@app.route("/")
def index():
    """
    Displays a dropdown list of training runs.
    """
    runs = [
        d
        for d in os.listdir(TRAIN_OUTPUTS_DIR)
        if os.path.isdir(os.path.join(TRAIN_OUTPUTS_DIR, d))
    ]
    runs.sort(
        key=lambda d: os.path.getctime(os.path.join(TRAIN_OUTPUTS_DIR, d)),
        reverse=True,
    )
    return render_template("index.html", runs=runs)


@app.route("/visualize/<run>")
def visualize(run):
    """
    Displays the results from all_results.json and the generated text data.
    """
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    if not os.path.isdir(logdir):
        abort(404, description=f"Run '{run}' not found.")

    eval_folders = [
        d
        for d in os.listdir(logdir)
        if os.path.isdir(os.path.join(logdir, d)) and d.startswith("eval-results-")
    ]
    eval_folders.sort(
        key=lambda d: os.path.getctime(os.path.join(logdir, d)),
        reverse=True,
    )

    # Get available checkpoints
    checkpoints = get_available_checkpoints(run)
    checkpoint_names = [os.path.basename(cp) for cp in checkpoints]

    selected_eval_folder = request.args.get("eval_folder")

    if eval_folders and selected_eval_folder:
        # New structure with eval folders and a folder is selected
        eval_path = os.path.join(logdir, selected_eval_folder)
        data = get_run_data(eval_path)
        if data is None:
            abort(
                404,
                description=f"'all_results.json' not found in '{selected_eval_folder}'.",
            )
        generated_data = get_generated_text_data(eval_path)
    elif not eval_folders:
        # Old structure: eval results in the root folder
        data = get_run_data(logdir)
        if data is None:
            abort(404, description=f"'all_results.json' not found in '{run}'.")
        generated_data = get_generated_text_data(logdir)
        selected_eval_folder = "root"  # Indicate root folder for template
    else:
        # New structure, but no eval folder selected yet
        data = None
        generated_data = None

    # Load config.yaml from the run directory (root level)
    config_path = os.path.join(logdir, "config.yaml")
    try:
        with open(config_path) as f:
            yaml.add_constructor("!", lambda loader, node: None)
            config = yaml.safe_load(f)
            model_name = config.get("model_name_or_path")
    except FileNotFoundError:
        model_name = None

    return render_template(
        "visualize.html",
        run=run,
        eval_folders=eval_folders,
        selected_eval_folder=selected_eval_folder,
        data=data,
        generated_data=generated_data,
        model_name=model_name,
        checkpoints=checkpoint_names,
    )


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    Loads the model and creates the pipeline.
    """
    global chat_generator
    global chat_model_name
    global chat_history
    global modulated_model

    print("Loading model...")
    # Reset chat history
    chat_history = [{"role": "system", "content": ""}]
    # Reset modulated model if it exists
    modulated_model = None

    run = request.form["run"]
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    config_path = os.path.join(logdir, "args.yaml")
    try:
        with open(config_path) as f:
            yaml.add_constructor("!", lambda loader, node: None)
            yaml.add_multi_constructor(
                "tag:yaml.org,2002:python/object",
                lambda loader, suffix, node: None,
                Loader=yaml.SafeLoader,
            )
            config = yaml.safe_load(f)
            chat_model_name = config.get("model_name_or_path")
    except FileNotFoundError:
        chat_model_name = None

    if chat_model_name:
        tokenizer = get_tokenizer(chat_model_name, train=False)

        # Handle system roles by loading a custom chat template if needed
        load_custom_chat_template(tokenizer, chat_model_name)

        print(f"Using model: {chat_model_name}")
        print(
            f"Tokenizer chat template available: {hasattr(tokenizer, 'chat_template')}"
        )

        chat_generator = pipeline(
            "text-generation", model=chat_model_name, tokenizer=tokenizer, device=device
        )
        return jsonify({"model_name": chat_model_name})
    else:
        return jsonify({"error": "Model name not found in args.yaml."})


@app.route("/load_checkpoint", methods=["POST"])
def load_checkpoint():
    """
    Loads a checkpoint and creates a modulated model.
    """
    global modulated_model
    global chat_history
    global chat_generator

    # Reset the chat history
    chat_history = [{"role": "system", "content": ""}]
    # Unload the regular chat model to avoid confusion
    chat_generator = None

    from ctx_to_lora.modeling_utils import ModulatedPretrainedModel

    run = request.form["run"]
    checkpoint = request.form["checkpoint"]
    contexts = request.form.getlist("contexts[]")

    # Filter out empty contexts
    contexts = [ctx for ctx in contexts if ctx.strip()]

    # If no contexts provided, use a single empty string
    if not contexts:
        contexts = [""]

    print(f"Received {len(contexts)} non-empty contexts for LoRA generation")

    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    checkpoint_path = os.path.join(logdir, checkpoint, "pytorch_model.bin")

    try:
        print(f"Loading checkpoint: {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, weights_only=False)

        modulated_model = ModulatedPretrainedModel.from_state_dict(
            state_dict,
            train=False,
            use_flash_attn=True,
        )
        modulated_model = modulated_model.to(device).to(torch.bfloat16)
        modulated_model.eval()

        result = {
            "success": True,
            "message": f"Loaded checkpoint {checkpoint}",
            "contexts_processed": len(contexts),
        }

        return jsonify(result)
    except Exception as e:
        print(f"Error loading checkpoint: {str(e)}")
        import traceback

        traceback.print_exc()
        return jsonify({"success": False, "error": str(e)})


def process_multiple_contexts(contexts, ctx_tokenizer):
    """
    Process multiple contexts for the ModulatedPretrainedModel.

    Args:
        contexts: List of context strings
        ctx_tokenizer: The tokenizer for the context model
        max_length: Maximum length for tokenization

    Returns:
        Dictionary with ctx_ids and ctx_attn_mask as tensors
    """
    # Remove empty contexts
    contexts = [ctx for ctx in contexts if ctx.strip()]

    # If no contexts provided, use a single empty string
    if not contexts:
        contexts = [""]

    print(f"Processing {len(contexts)} non-empty contexts")

    tokenized_contexts = tokenize_ctx_text({"context": contexts}, ctx_tokenizer)
    ctx_ids = tokenized_contexts["ctx_ids"]
    ctx_attn_mask = tokenized_contexts["ctx_attn_mask"]
    ctx_ids = torch.nn.utils.rnn.pad_sequence(
        torch.tensor(ctx_ids, dtype=torch.long, device=device),
        batch_first=True,
        padding_value=0,
    )
    ctx_attn_mask = torch.nn.utils.rnn.pad_sequence(
        torch.tensor(ctx_attn_mask, dtype=torch.long, device=device),
        batch_first=True,
        padding_value=0,
    )

    return {"ctx_ids": ctx_ids, "ctx_attn_mask": ctx_attn_mask}


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles chat requests and generates responses.
    """
    global chat_history
    global modulated_model
    print("Chat request received")
    message = request.form["message"]
    print(f"Received message: {message}")
    if chat_history is None:
        chat_history = [{"role": "system", "content": ""}]

    chat_history.append({"role": "user", "content": message})

    if chat_generator or modulated_model:
        print("Generating response...")
        try:
            # Handle both regular chat_generator and modulated_model
            if modulated_model:
                print("Using modulated model for response generation")
                # Process context and generate LoRA for the response
                ctx_tokenizer = None
                with torch.inference_mode(), torch.amp.autocast(str(device)):
                    from transformers import AutoTokenizer

                    # Get the contexts and tokenize them
                    contexts = request.form.getlist("contexts[]")
                    if not contexts:
                        contexts = [""]  # Use empty context if none provided

                    # Ensure we have at least one context
                    if len(contexts) == 1 and not contexts[0].strip():
                        contexts = [""]

                    print(f"Processing {len(contexts)} contexts for response generation")
                    print(f"Contexts: {contexts}")
                    ctx_encoder_model_name_or_path = (
                        modulated_model.ctx_encoder_args.ctx_encoder_model_name_or_path
                        or modulated_model.base_model.config.name_or_path
                    )
                    ctx_tokenizer = get_tokenizer(
                        ctx_encoder_model_name_or_path, train=False
                    )

                    # Prepare base model inputs
                    base_tokenizer = get_tokenizer(
                        modulated_model.base_model.config.name_or_path, train=False
                    )

                    # Process the contexts for the ctx_encoder
                    ctx_inputs = process_multiple_contexts(
                        contexts,
                        ctx_tokenizer,
                        # max_length=(
                        #     modulated_model.ctx_encoder_args.max_ctx_len
                        #     if hasattr(modulated_model.ctx_encoder_args, "max_ctx_len")
                        #     else 512
                        # ),
                    )

                    ctx_ids = ctx_inputs["ctx_ids"].to(device)
                    ctx_attn_mask = ctx_inputs["ctx_attn_mask"].to(device)

                    print(f"chat_history: {chat_history}")

                    # Tokenize the chat history
                    model_inputs = base_tokenizer.apply_chat_template(
                        chat_history, return_tensors="pt", add_generation_prompt=True
                    ).to(device)

                    # Generate response with context-modulated model
                    # with modulated_model.generate_and_apply_loras(
                    #     ctx_ids=ctx_ids,
                    #     ctx_attn_mask=ctx_attn_mask,
                    # ) as applied_model:
                    #     outputs = applied_model.generate(
                    #         input_ids=model_inputs,
                    #         max_new_tokens=512,
                    #         do_sample=False,
                    #     )

                    outputs = modulated_model.generate_with_multi_loras(
                        ctx_ids=ctx_ids,
                        ctx_attn_mask=ctx_attn_mask,
                        input_ids=model_inputs,
                        max_new_tokens=512,
                        do_sample=False,
                    )

                    # outputs = modulated_model.generate(
                    #     ctx_ids=ctx_ids,
                    #     ctx_attn_mask=ctx_attn_mask,
                    #     input_ids=model_inputs,
                    #     max_new_tokens=512,
                    #     do_sample=False,
                    # )

                    # Decode the generated response
                    response = base_tokenizer.decode(
                        outputs[0][model_inputs.shape[1] :], skip_special_tokens=True
                    )

                print(f"Modulated model response: {response}")
            elif chat_generator:
                response = chat_generator(chat_history, max_length=2**13)[0][
                    "generated_text"
                ][-1]["content"]
            else:
                response = "No model loaded. Please load a model first."

            print(f"Response: {response}")
            chat_history.append({"role": "assistant", "content": response})
            return jsonify({"response": response})
        except Exception as e:
            print(f"Error generating response: {str(e)}")
            import traceback

            traceback.print_exc()
            return jsonify({"response": f"Error: {str(e)}"})
    else:
        return jsonify(
            {
                "response": "No model loaded. Please load a base model or apply a hypernetwork first."
            }
        )


@app.route("/update_system_msg", methods=["POST"])
def update_system_msg():
    """
    Updates the system message in chat_history.
    """
    global chat_history
    data = request.get_json()
    new_system_msg = data.get("system_msg", "").strip()
    if new_system_msg:
        if chat_history and chat_history[0]["content"] != new_system_msg:
            chat_history[0]["content"] = new_system_msg
            return jsonify({"success": True})
    return jsonify({"success": False})


@app.route("/get_system_msg", methods=["GET"])
def get_system_msg():
    """
    Retrieves the current system message from chat_history.
    """
    global chat_history
    if chat_history and "content" in chat_history[0]:
        return jsonify({"system_msg": chat_history[0]["content"]})
    return jsonify({"system_msg": ""})


@app.route("/get_command/<run>")
def get_command(run):
    """
    Extracts the command from the debug.log file.

    Args:
        run: The name of the training run.

    Returns:
        A JSON response containing the command or an error message.
    """
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    debug_log_path = os.path.join(logdir, "debug.log")

    try:
        with open(debug_log_path) as f:
            for line in f:
                if "CMD:" in line:
                    command = line.split("CMD:")[1].strip()
                    return jsonify({"command": command})
        return jsonify({"command": "Command not found in debug.log."})
    except FileNotFoundError:
        return jsonify({"command": "debug.log not found."})


@app.route("/get_config/<run>")
def get_config(run):
    """
    Extracts the config file name from cli_args.yaml.

    Args:
        run: The name of the training run.

    Returns:
        A JSON response containing the config file name or an error message.
    """
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    cli_args_path = os.path.join(logdir, "cli_args.yaml")

    try:
        with open(cli_args_path) as f:
            cli_args = yaml.safe_load(f)
            config_file = cli_args.get("config", "Config file not found.")
            return jsonify({"config": config_file})
    except FileNotFoundError:
        return jsonify({"config": "cli_args.yaml not found."})


@app.route("/reset_chat", methods=["POST"])
def reset_chat():
    """
    Resets the chat history to start a new conversation.
    """
    global chat_history

    print("Resetting chat history")
    chat_history = [{"role": "system", "content": ""}]

    # Get the system message if it was set previously
    system_msg = request.form.get("system_msg", "")
    if system_msg:
        chat_history[0]["content"] = system_msg

    return jsonify({"success": True, "message": "Chat history reset successfully"})


if __name__ == "__main__":
    app.run(debug=True)
