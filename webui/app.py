import os
import json
from flask import Flask, render_template, request, abort

app = Flask(__name__)

TRAIN_OUTPUTS_DIR = "train_outputs"


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
        with open(results_file, "r") as f:
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
    Loads generated text data from test_generated_text.jsonl and val_generated_text.jsonl.

    Args:
        run_path: Path to the directory containing the .jsonl files.

    Returns:
        A dictionary containing the generated text data for "test" and "val".
    """
    generated_data = {}
    for split in ["test", "val"]:
        filename = f"{split}_generated_text.jsonl"
        filepath = os.path.join(run_path, filename)
        try:
            with open(filepath, "r") as f:
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

    return render_template(
        "visualize.html", run=run, data=data, generated_data=generated_data
    )


if __name__ == "__main__":
    app.run(debug=True)
