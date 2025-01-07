import json
import os
import re
from glob import glob

import torch
import yaml
from flask import Flask, abort, jsonify, render_template, request
from transformers import pipeline

app = Flask(__name__)

TRAIN_OUTPUTS_DIR = "train_outputs"
chat_generator = None
chat_model_name = None
chat_history = None
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def get_run_data(run_path):
    """
    Loads and processes data from all_results.json for a given run.

    Args:
        run_path: Path to the directory containing the all_results.json file.

    Returns:
        A dictionary containing the grouped results data.
    """
    results_file = os.path.join(run_path, "all_results.json")
    try:
        with open(results_file) as f:
            data = json.load(f)
    except FileNotFoundError:
        return None

    grouped_data = {}
    for key, value in data.items():
        if isinstance(value, float):
            value = round(value, 4)
        prefix = key.split("_")[0]
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

    data = get_run_data(logdir)
    if data is None:
        abort(404, description=f"'all_results.json' not found in '{run}'.")

    generated_data = get_generated_text_data(logdir)

    # Load config.yaml from the run directory
    config_path = os.path.join(logdir, "config.yaml")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            model_name = config.get("model_name_or_path")
    except FileNotFoundError:
        model_name = None

    return render_template(
        "visualize.html",
        run=run,
        data=data,
        generated_data=generated_data,
        model_name=model_name,
    )


@app.route("/load_model", methods=["POST"])
def load_model():
    """
    Loads the model and creates the pipeline.
    """
    global chat_generator
    global chat_model_name
    global chat_history

    print("Loading model...")
    chat_history = [{"role": "system", "content": ""}]
    run = request.form["run"]
    logdir = os.path.join(TRAIN_OUTPUTS_DIR, run)
    config_path = os.path.join(logdir, "config.yaml")
    try:
        with open(config_path) as f:
            config = yaml.safe_load(f)
            chat_model_name = config.get("model_name_or_path")
    except FileNotFoundError:
        chat_model_name = None

    if chat_model_name:
        chat_generator = pipeline(
            "text-generation", model=chat_model_name, device=device
        )
        return jsonify({"model_name": chat_model_name})
    else:
        return jsonify({"error": "Model name not found in config.yaml."})


@app.route("/chat", methods=["POST"])
def chat():
    """
    Handles chat requests and generates responses.
    """
    global chat_history
    print("Chat request received")
    message = request.form["message"]
    print(f"Received message: {message}")
    chat_history.append({"role": "user", "content": message})

    if chat_generator:
        print("Generating response...")
        response = chat_generator(chat_history, max_length=2**13)[0]["generated_text"][
            -1
        ]["content"]
        print(f"Response: {response}")
    else:
        response = "Chat model not loaded."
    return jsonify({"response": response})


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


if __name__ == "__main__":
    app.run(debug=True)
