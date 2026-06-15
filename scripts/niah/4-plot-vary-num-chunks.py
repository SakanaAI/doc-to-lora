from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_ablation_utils import (
    choose_latest_runs_by_dataset_and_chunk,
    dataset_context_length,
    ensure_output_path,
    get_dense_values,
    get_performance_value,
    load_script_text,
    parse_checkpoint_path,
    parse_eval_results_root,
    parse_expected_datasets,
)

LATTE_STYLE = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"


def configure_plot_style() -> None:
    plt.style.use(["ggplot", LATTE_STYLE])
    plt.rcParams["axes.facecolor"] = "F7FBFC"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["grid.color"] = "cccccc"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "ccd0da"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.fontsize"] = 14
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=plt.cm.tab10.colors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot the num-chunks ablation from 4-eval-vary-num-chunks.sh.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).with_name("4-eval-vary-num-chunks.sh"),
        help="Path to the eval sweep shell script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to eval-results-*/plots/4-vary-num-chunks.png.",
    )
    parser.add_argument(
        "--metric",
        default="rougeL.f1",
        help="Performance metric to read from each *_results.json file.",
    )
    parser.add_argument(
        "--dense-metric",
        default="mean",
        help="Dense LoRA metric to read when available.",
    )
    parser.add_argument(
        "--dense-std-metric",
        default="std",
        help="Dense LoRA std metric to use for the shaded area when available.",
    )
    parser.add_argument(
        "--dense-max-metric",
        default="max",
        help="Dense LoRA max metric to plot as a separate line when available.",
    )
    return parser


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()

    script_text = load_script_text(args.eval_script)
    checkpoint_path = parse_checkpoint_path(script_text)
    eval_results_root = parse_eval_results_root(checkpoint_path)
    expected_datasets = parse_expected_datasets(script_text)
    selected_runs_by_dataset = choose_latest_runs_by_dataset_and_chunk(
        eval_results_root,
        expected_datasets,
    )

    if not any(selected_runs_by_dataset.values()):
        raise SystemExit(f"No matching eval runs found under {eval_results_root}.")

    output_path = (
        eval_results_root / "plots" / "4-vary-num-chunks.png"
        if args.output is None
        else args.output
    )
    output_path = ensure_output_path(output_path)

    fig, (ax_perf, ax_dense, ax_dense_max) = plt.subplots(
        1,
        3,
        figsize=(20, 5.5),
        constrained_layout=False,
    )
    fig.subplots_adjust(bottom=0.27, top=0.88, wspace=0.28)

    color_map = plt.get_cmap("tab10", len(expected_datasets))
    all_num_chunks: set[int] = set()
    has_any_dense_mean_values = False
    has_any_dense_max_values = False

    for index, dataset_name in enumerate(expected_datasets):
        color = color_map(index)
        context_length = dataset_context_length(dataset_name)
        label = rf"$2^{{{int(context_length).bit_length() - 1}}}$"

        num_chunks: list[int] = []
        performance_values: list[float] = []
        dense_mean_xs: list[int] = []
        dense_mean_ys: list[float] = []
        dense_max_xs: list[int] = []
        dense_max_ys: list[float] = []

        for run_info in selected_runs_by_dataset.get(dataset_name, []):
            if run_info.chunk_size == 24:
                continue

            approx_num_chunks = max(1, context_length // run_info.chunk_size)
            performance = get_performance_value(
                run_info.run_dir,
                dataset_name,
                metric_name=args.metric,
            )
            if performance is None:
                continue

            num_chunks.append(approx_num_chunks)
            performance_values.append(performance)
            all_num_chunks.add(approx_num_chunks)

            dense_stats = get_dense_values(
                run_info.run_dir,
                dataset_name,
                metric_names=(
                    args.dense_metric,
                    args.dense_std_metric,
                    args.dense_max_metric,
                ),
            )
            dense_mean = dense_stats[args.dense_metric]
            dense_max = dense_stats[args.dense_max_metric]
            if dense_mean is not None:
                dense_mean_xs.append(approx_num_chunks)
                dense_mean_ys.append(dense_mean)
            if dense_max is not None:
                dense_max_xs.append(approx_num_chunks)
                dense_max_ys.append(dense_max)

        if performance_values:
            ax_perf.plot(
                num_chunks,
                performance_values,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
                markeredgecolor="black",
                markeredgewidth=1,
            )
        if dense_mean_ys:
            has_any_dense_mean_values = True
            ax_dense.plot(
                dense_mean_xs,
                dense_mean_ys,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
                markeredgecolor="black",
                markeredgewidth=1,
            )
        if dense_max_ys:
            has_any_dense_max_values = True
            ax_dense_max.plot(
                dense_max_xs,
                dense_max_ys,
                marker="o",
                linewidth=2,
                color=color,
                label=label,
                markeredgecolor="black",
                markeredgewidth=1,
            )

    if not all_num_chunks:
        raise SystemExit("No performance values were found for the selected runs.")

    sorted_num_chunks = sorted(all_num_chunks)
    for axis in (ax_perf, ax_dense, ax_dense_max):
        axis.set_xscale("log", base=2)
        axis.set_xticks(sorted_num_chunks)
        axis.set_xticklabels(
            [
                rf"$2^{{{math.log2(value):.1f}}}$"
                if not math.log2(value).is_integer()
                else rf"$2^{{{int(math.log2(value))}}}$"
                for value in sorted_num_chunks
            ],
            rotation=0,
            ha="center",
        )
        axis.grid(True, alpha=0.25)
        axis.xaxis.set_minor_locator(plt.NullLocator())

    ax_perf.set_title("Retrieval Performance", fontweight="bold")
    ax_perf.set_xlabel("Number of Chunks")
    ax_perf.set_ylabel("ROUGE-L F1 Score")
    ax_perf.set_ylim(-0.05, 1.05)

    ax_dense.set_title("LoRA Mean Magnitude", fontweight="bold")
    ax_dense.set_xlabel("Number of Chunks")
    ax_dense.set_ylabel("Mean Per-Element Magnitude")
    if has_any_dense_mean_values:
        ax_dense.set_ylim(bottom=0.0)
    else:
        ax_dense.text(
            0.5,
            0.5,
            "Dense LoRA stats are missing for these runs.",
            ha="center",
            va="center",
            transform=ax_dense.transAxes,
        )

    ax_dense_max.set_title("LoRA Max Magnitude", fontweight="bold")
    ax_dense_max.set_xlabel("Number of Chunks")
    ax_dense_max.set_ylabel("Maximum Per-Element Magnitude")
    if has_any_dense_max_values:
        ax_dense_max.set_ylim(bottom=0.0)
    else:
        ax_dense_max.text(
            0.5,
            0.5,
            "Dense LoRA max stats are missing for these runs.",
            ha="center",
            va="center",
            transform=ax_dense_max.transAxes,
        )

    handles, labels = ax_perf.get_legend_handles_labels()
    haystack_legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.005),
        ncol=min(6, len(labels)),
        frameon=True,
        fancybox=True,
        title="Haystack length",
        title_fontsize=16,
    )
    haystack_legend.get_frame().set_edgecolor("#bdbdbd")
    haystack_legend.get_frame().set_linewidth(1.0)
    haystack_legend.get_frame().set_alpha(1.0)

    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
