from __future__ import annotations

import argparse
import math
from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plot_ablation_utils import (
    choose_latest_runs_by_dataset_chunk_and_overlap,
    dataset_context_length,
    ensure_output_path,
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
        description="Compare overlapping vs non-overlapping chunking for the 4-* NIAH sweeps.",
    )
    parser.add_argument(
        "--non-overlap-script",
        type=Path,
        default=Path(__file__).with_name("4-eval-vary-num-chunks.sh"),
        help="Path to the non-overlapping eval sweep shell script.",
    )
    parser.add_argument(
        "--overlap-script",
        type=Path,
        default=Path(__file__).with_name("4-eval-vary-num-chunks-overlap.sh"),
        help="Path to the overlapping eval sweep shell script.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output PNG path. Defaults to eval-results-*/plots/4-vary-num-chunks-overlap-comparison.png.",
    )
    parser.add_argument(
        "--metric",
        default="rougeL.f1",
        help="Performance metric to read from each *_results.json file.",
    )
    parser.add_argument(
        "--non-overlap-value",
        type=int,
        default=0,
        help="Overlap value used for the non-overlapping sweep.",
    )
    parser.add_argument(
        "--overlap-value",
        type=int,
        default=8,
        help="Overlap value used for the overlapping sweep.",
    )
    return parser


def tick_label(chunk_size: int) -> str:
    exponent = math.log2(chunk_size)
    if exponent.is_integer():
        return rf"$2^{{{int(exponent)}}}$"
    return rf"$2^{{{exponent:.1f}}}$"


def dataset_title(dataset_name: str) -> str:
    context_length = dataset_context_length(dataset_name)
    exponent = int(math.log2(context_length))
    return rf"Haystack Length $2^{{{exponent}}}$"


def line_label(overlap_value: int) -> str:
    if overlap_value == 0:
        return "Non-overlapping"
    return f"Overlap = {overlap_value}"


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()

    non_overlap_text = load_script_text(args.non_overlap_script)
    overlap_text = load_script_text(args.overlap_script)

    checkpoint_path = parse_checkpoint_path(non_overlap_text)
    eval_results_root = parse_eval_results_root(checkpoint_path)
    overlap_eval_results_root = parse_eval_results_root(
        parse_checkpoint_path(overlap_text)
    )
    if overlap_eval_results_root != eval_results_root:
        raise SystemExit(
            "The overlap and non-overlap scripts point to different eval-results roots."
        )

    non_overlap_datasets = set(parse_expected_datasets(non_overlap_text))
    overlap_datasets = set(parse_expected_datasets(overlap_text))
    expected_datasets = sorted(
        non_overlap_datasets & overlap_datasets,
        key=dataset_context_length,
    )
    if not expected_datasets:
        raise SystemExit("No shared datasets were found between the two sweep scripts.")

    non_overlap_runs = choose_latest_runs_by_dataset_chunk_and_overlap(
        eval_results_root,
        expected_datasets,
        ctx_chunk_overlap=args.non_overlap_value,
    )
    overlap_runs = choose_latest_runs_by_dataset_chunk_and_overlap(
        eval_results_root,
        expected_datasets,
        ctx_chunk_overlap=args.overlap_value,
    )

    output_path = (
        eval_results_root / "plots" / "4-vary-num-chunks-overlap-comparison.png"
        if args.output is None
        else args.output
    )
    output_path = ensure_output_path(output_path)

    num_datasets = len(expected_datasets)
    fig, axes = plt.subplots(
        1,
        num_datasets,
        figsize=(6.4 * num_datasets, 5.5),
        squeeze=False,
        constrained_layout=False,
    )
    fig.subplots_adjust(bottom=0.26, top=0.84, wspace=0.28)

    color_map = plt.get_cmap("tab10", 2)
    legend_handles = []
    legend_labels = []

    for axis, dataset_name in zip(axes[0], expected_datasets):
        plotted_anything = False
        dataset_chunk_sizes: set[int] = set()

        for overlap_value, run_infos, color in (
            (
                args.non_overlap_value,
                non_overlap_runs.get(dataset_name, []),
                color_map(0),
            ),
            (args.overlap_value, overlap_runs.get(dataset_name, []), color_map(1)),
        ):
            chunk_sizes: list[int] = []
            performance_values: list[float] = []
            for run_info in run_infos:
                performance = get_performance_value(
                    run_info.run_dir,
                    dataset_name,
                    metric_name=args.metric,
                )
                if performance is None:
                    continue
                chunk_sizes.append(run_info.chunk_size)
                performance_values.append(performance)
                dataset_chunk_sizes.add(run_info.chunk_size)

            if not performance_values:
                continue

            plotted_anything = True
            (line,) = axis.plot(
                chunk_sizes,
                performance_values,
                marker="o",
                linewidth=2,
                color=color,
                label=line_label(overlap_value),
                markeredgecolor="black",
                markeredgewidth=1,
            )
            if line.get_label() not in legend_labels:
                legend_handles.append(line)
                legend_labels.append(line.get_label())

        if not plotted_anything:
            axis.text(
                0.5,
                0.5,
                "No matching runs found.",
                ha="center",
                va="center",
                transform=axis.transAxes,
            )
        else:
            sorted_chunk_sizes = sorted(dataset_chunk_sizes)
            axis.set_xscale("log", base=2)
            axis.set_xticks(sorted_chunk_sizes)
            axis.set_xticklabels([tick_label(value) for value in sorted_chunk_sizes])
            axis.xaxis.set_minor_locator(plt.NullLocator())

        axis.grid(True, alpha=0.25)
        axis.set_title(dataset_title(dataset_name), fontweight="bold")
        axis.set_xlabel("Chunk Size")
        axis.set_ylabel("ROUGE-L F1 Score")
        axis.set_ylim(-0.05, 1.05)

    if legend_handles:
        legend = fig.legend(
            legend_handles,
            legend_labels,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.01),
            ncol=len(legend_labels),
            frameon=True,
            fancybox=True,
        )
        legend.get_frame().set_edgecolor("#bdbdbd")
        legend.get_frame().set_linewidth(1.0)
        legend.get_frame().set_alpha(1.0)

    fig.suptitle("Overlapping vs Non-overlapping Chunking", fontweight="bold")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
