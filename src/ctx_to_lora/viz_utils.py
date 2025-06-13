from glob import glob

import pandas as pd

# convert to llm-comparator format
"""{
    "metadata": {
        "source_path": "Any string for your records (e.g., run id)",
        "custom_fields_schema": []
    },
    "models": [
        {"name": "Short name of your first model"},
        {"name": "Short name of your second model"}
    ],
    "examples": [
        {
            "input_text": "This is a prompt.",
            "tags": ["Math"],  # A list of keywords for categorizing prompts
            "output_text_a": "Response to the prompt from the first model (A)",
            "output_text_b": "Response to the prompt from the other model (B)",
            "score": -1.25,  # Score from the judge LLM
            "individual_rater_scores": [],
            "custom_fields": {}
        },
        {
            "input_text": "This is a next prompt.",
            ...
        }
    ]
}
"""


def convert_to_llm_comparator_format(
    eval_checkpoint_dir: str,
    model_dir: str,
    compare_to_no_context: bool = False,
) -> dict:
    # return {
    #     "metadata": {"source_path": source_path, "custom_fields_schema": []},
    #     "models": [{"name": model_a_name}, {"name": model_b_name}],
    #     "examples": examples,
    # }

    checkpoint_name = "/".join(eval_checkpoint_dir.strip("/").split("/")[-2:])
    files_checkpoint = sorted(
        glob(f"{eval_checkpoint_dir}/**/*generated*.jsonl", recursive=True)
    )
    files_checkpoint = {file.split("/")[-1]: file for file in files_checkpoint}

    eval_model_dir = f"eval_results/{model_dir}"
    if compare_to_no_context:
        files_model = sorted(
            glob(f"{eval_model_dir}/**/*no_context*generated*.jsonl", recursive=True)
        )
    else:
        files_model = sorted(
            glob(f"{eval_model_dir}/**/*[!no_context]*generated*.jsonl", recursive=True)
        )
    files_model = {file.split("/")[-1]: file for file in files_model}

    overlap_files = set(files_checkpoint) & set(files_model)
    if not overlap_files:
        raise ValueError(
            "No overlapping files found between the evaluation checkpoint and model directory.\n"
            f"Checkpoint files: {files_checkpoint.keys()}\n"
            f"Model files: {files_model.keys()}"
        )

    print(
        f"Generating results from {len(overlap_files)} overlapping files.\n"
        + f"{overlap_files}"
    )

    out = {
        "metadata": {
            "source_path": checkpoint_name,
            "custom_fields_schema": [
                {"name": "question", "type": "text"},
                {"name": "label", "type": "text"},
                {"name": "is_base_model_correct", "type": "category"},
                {"name": "is_checkpoint_model_correct", "type": "category"},
                {"name": "ctx_ids_len", "type": "number"},
                {"name": "query_ids_len", "type": "number"},
            ],
        },
        "models": [{"name": checkpoint_name}, {"name": model_dir}],
        "examples": [],
    }
    for file in overlap_files:
        df_checkpoint = pd.read_json(path_or_buf=files_checkpoint[file], lines=True)
        print(df_checkpoint.columns)
        score_name = [col for col in df_checkpoint.columns if col.endswith("_score")]
        assert len(score_name) == 1, (
            "Expected exactly one score column. Got: {score_name}"
        )

        df_model = pd.read_json(path_or_buf=files_model[file], lines=True)
        print(df_model)
        score_name = [col for col in df_model.columns if col.endswith("_score")]
        assert len(score_name) == 1, (
            "Expected exactly one score column. Got: {score_name}"
        )

        score_name = score_name[0]

        # df_checkpoint.set_index("input", inplace=True)
        df_checkpoint.drop(columns=["label"], inplace=True)
        # df_model.set_index("input", inplace=True)
        df = df_checkpoint.join(df_model, lsuffix="_checkpoint", rsuffix="_model")
        # print(df.head())
        # print(df.tail())
        # print(df.columns)
        # sample = df.iloc[0]
        # print(sample["context"])
        # print(sample["input_checkpoint"])
        # print()
        # print(sample["input_model"])
        df.drop(columns=["input_model"], inplace=True)
        # print(f"{sample['generated_checkpoint']=}")
        # print(f"{sample['generated_model']=}")

        for sample in df.itertuples():
            score_a = getattr(sample, f"{score_name}_checkpoint")
            score_b = getattr(sample, f"{score_name}_model")
            out["examples"].append(
                {
                    "input_text": sample.context,
                    "tags": [file],
                    "output_text_a": sample.generated_checkpoint,
                    "output_text_b": sample.generated_model,
                    "score": score_a - score_b,
                    "individual_rater_scores": [],
                    "custom_fields": {
                        "question": sample.input_checkpoint,
                        "label": sample.label,
                        "is_base_model_correct": score_b >= 0.5,
                        "is_checkpoint_model_correct": score_a >= 0.5,
                        "ctx_ids_len": sample.ctx_ids_len,
                        "query_ids_len": sample.input_ids_len_checkpoint,
                    },
                }
            )

        # break
        # # print(json_checkpoint[0])
        # # print()
        # # print(json_model[0])
    return out


def build_confusion_matrix(
    comparator_json: dict,
    model_a_name: str = "checkpoint",
    model_b_name: str = "model",
) -> pd.DataFrame:
    """
    Build a confusion matrix from the comparator JSON.
    """
    examples = comparator_json["examples"]
    df = pd.DataFrame(examples)
    df["is_correct_a"] = df["custom_fields"].apply(
        lambda x: x.get("is_checkpoint_model_correct")
    )
    df["is_correct_b"] = df["custom_fields"].apply(
        lambda x: x.get("is_base_model_correct")
    )

    confusion_matrix = pd.crosstab(
        df["is_correct_a"],
        df["is_correct_b"],
        rownames=[model_a_name],
        colnames=[model_b_name],
        margins=True,
        margins_name="Total",
    )
    return confusion_matrix


if __name__ == "__main__":
    # Example usage
    import json

    eval_checkpoint_dir = "train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/eval-results-105000"
    model_dir = "google/gemma-2-2b-it"
    comparator_json = convert_to_llm_comparator_format(eval_checkpoint_dir, model_dir)
    confusion_matrix = build_confusion_matrix(comparator_json)
    print(confusion_matrix)
    # Save the comparator JSON to a file
    with open(f"{eval_checkpoint_dir}/comparator.json", "w") as f:
        json.dump(comparator_json, f)
