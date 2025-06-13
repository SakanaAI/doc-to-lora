from glob import glob

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap


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
                        "is_base_model_correct": score_b >= 0.4,
                        "is_checkpoint_model_correct": score_a >= 0.4,
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
    model_a_name: str = "Adapted model",
    model_b_name: str = "Base model",
) -> pd.DataFrame:
    """
    Build a confusion matrix from the comparator JSON.
    """
    examples = comparator_json["examples"]
    df = pd.DataFrame(examples)
    df["is_correct_a"] = df["custom_fields"].apply(
        lambda x: x.get("is_checkpoint_model_correct")
    )
    df["is_correct_a"] = pd.Categorical(
        df["is_correct_a"], categories=[True, False], ordered=True
    )
    df["is_correct_b"] = df["custom_fields"].apply(
        lambda x: x.get("is_base_model_correct")
    )
    df["is_correct_b"] = pd.Categorical(
        df["is_correct_b"], categories=[True, False], ordered=True
    )

    confusion_matrix = pd.crosstab(
        df["is_correct_a"],
        df["is_correct_b"],
        rownames=[model_a_name],
        colnames=[model_b_name],
        margins=True,
        margins_name="Total",
        dropna=False,
        normalize="all",
    )
    return confusion_matrix


def plot_confusion_matrix(
    confusion_matrix: pd.DataFrame,
    title: str = "Confusion Matrix",
    save_path: str | None = None,
) -> plt.Figure:
    """
    Plot a confusion matrix using matplotlib with a soothing color palette.

    Args:
        confusion_matrix: DataFrame with confusion matrix data
        title: Title for the plot
        save_path: Path to save the plot (optional)

    Returns:
        Matplotlib Figure object
    """
    # Remove the 'Total' row and column for visualization
    viz_matrix = confusion_matrix.iloc[:-1, :-1]

    # Set up the figure with a clean, modern style
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(8, 6))

    # Create a soothing color palette (soft blues to deeper blues)
    colors = ["#f0f8ff", "#e6f3ff", "#b3d9ff", "#4d94ff", "#0066cc"]
    soothing_cmap = LinearSegmentedColormap.from_list("soothing_blues", colors)

    # Create the heatmap
    im = ax.imshow(viz_matrix.values, interpolation="nearest", cmap=soothing_cmap)

    # Add colorbar with proper formatting
    cbar = plt.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("Proportion", rotation=270, labelpad=20, fontsize=12)

    # Set ticks and labels
    ax.set_xticks(np.arange(len(viz_matrix.columns)))
    ax.set_yticks(np.arange(len(viz_matrix.index)))
    ax.set_xticklabels(viz_matrix.columns, fontsize=11)
    ax.set_yticklabels(viz_matrix.index, fontsize=11)

    # Add text annotations with values
    for i in range(len(viz_matrix.index)):
        for j in range(len(viz_matrix.columns)):
            text = ax.text(
                j,
                i,
                f"{viz_matrix.iloc[i, j]:.3f}",
                ha="center",
                va="center",
                color="black",
                fontsize=12,
                fontweight="bold",
            )

    # Set labels and title
    ax.set_xlabel(
        f"{viz_matrix.columns.name} (Correct)", fontsize=13, fontweight="bold"
    )
    ax.set_ylabel(f"Adapted model (Correct)", fontsize=13, fontweight="bold")
    ax.set_title(title, fontsize=15, fontweight="bold", pad=20)

    # Calculate and add performance metrics annotations
    same_results = viz_matrix.loc[True, True] + viz_matrix.loc[False, False]

    # Calculate boost rate and resilience rate from the full confusion matrix
    full_matrix = confusion_matrix.iloc[:-1, :-1]  # Remove totals for calculations
    total_base_correct = confusion_matrix.loc["Total", True]
    boost_rate = (
        full_matrix.loc[True, False] / total_base_correct
        if total_base_correct > 0
        else 0
    )
    resilience_rate = (
        full_matrix.loc[True, True] / total_base_correct
        if total_base_correct > 0
        else 0
    )

    # Add metrics text box
    metrics_text = (
        f"Same results: {same_results * 100:.2f}%\n\n"
        f"Boost Rate: {boost_rate * 100:.2f}%\n"
        f"Resilience Rate: {resilience_rate * 100:.2f}%"
    )

    ax.text(
        0.02,
        0.98,
        metrics_text,
        transform=ax.transAxes,
        fontsize=11,
        fontweight="bold",
        bbox=dict(
            boxstyle="round,pad=0.4", facecolor="white", edgecolor="gray", alpha=0.9
        ),
        verticalalignment="top",
    )

    # Improve layout
    plt.tight_layout()

    if save_path:
        if save_path.endswith(".pdf"):
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        elif save_path.endswith(".png"):
            plt.savefig(save_path, format="png", dpi=300, bbox_inches="tight")
        else:
            # Default to PDF if no extension specified
            save_path = save_path + ".pdf"
            plt.savefig(save_path, format="pdf", dpi=300, bbox_inches="tight")
        print(f"Confusion matrix plot saved to: {save_path}")

    return fig


if __name__ == "__main__":
    # Example usage
    import json

    eval_checkpoint_dir = "train_outputs/runs/May08_13-56-31_slurm0-a3nodeset-5_59383_906acb28/eval-results-105000"
    model_dir = "google/gemma-2-2b-it"
    compare_to_no_context = False
    comparator_json = convert_to_llm_comparator_format(
        eval_checkpoint_dir, model_dir, compare_to_no_context
    )
    confusion_matrix = build_confusion_matrix(
        comparator_json,
        model_a_name="Adapted model",
        model_b_name="Base model no context" if compare_to_no_context else "Base model",
    )
    print(confusion_matrix)
    print(
        f"Boost rate: {confusion_matrix.loc[True, False] / confusion_matrix.loc['Total', True] * 100:.2f}%"
    )
    print(
        f"Resilience rate: {confusion_matrix.loc[True, True] / confusion_matrix.loc['Total', True] * 100:.2f}%"
    )

    # Plot and save confusion matrix
    checkpoint_name = "/".join(eval_checkpoint_dir.strip("/").split("/")[-2:])
    plot_title = f"Model Comparison: {checkpoint_name} vs {model_dir}"
    save_path = f"{eval_checkpoint_dir}/confusion_matrix.pdf"
    # TODO: add resilience + boost rate
    fig = plot_confusion_matrix(confusion_matrix, title=plot_title, save_path=save_path)
    # fig.show()

    # Save the comparator JSON to a file
    with open(f"{eval_checkpoint_dir}/comparator.json", "w") as f:
        json.dump(comparator_json, f)
