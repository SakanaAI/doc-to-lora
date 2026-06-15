from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append(str(Path(__file__).resolve().parents[2] / "niah"))

from plot_ablation_utils import (  # noqa: E402
    choose_latest_runs_by_chunk,
    ensure_output_path,
    get_performance_value,
    load_results_json,
    load_script_text,
    parse_checkpoint_path,
    parse_eval_results_root,
)

LATTE_STYLE = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"
DATASET_LABELS = {
    "longbench/qasper_e": "Qasper",
    "longbench/2wikimqa_e": "2WikiMQA",
    "longbench/multifieldqa_en_e": "MultiFieldQA-EN",
    "longbench/gov_report_e": "GovReport",
}
REFERENCE_PERFORMANCE = {
    "longbench/2wikimqa_e": 0.3387,
    "longbench/multifieldqa_en_e": 0.3938,
    "longbench/qasper_e": 0.3839,
}
METRIC_CANDIDATES = (
    "qa_f1_score",
    "rougeL.f1",
    "qa_f1",
    "f1",
    "accuracy",
)


def chunk_size_tick_label(chunk_size: int) -> str:
    return rf"$2^{{{int(chunk_size).bit_length() - 1}}}$"


def approx_chunk_label(value: float) -> str:
    rounded = round(value)
    if abs(value - rounded) < 0.05:
        return str(int(rounded))
    return f"{value:.1f}"


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
        description="Plot one D2L chunking-ablation figure per dataset.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).with_name("d2l-chunking-abalation.sh"),
        help="Path to the eval sweep shell script.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional directory for output PNGs. Defaults to eval-results-*/plots/.",
    )
    return parser


def parse_checkpoint_from_script(script_text: str) -> Path:
    try:
        return parse_checkpoint_path(script_text)
    except ValueError:
        match = re.search(r"--checkpoint_path\s+(\S+)", script_text)
        if match is None:
            raise
        return Path(match.group(1))


def infer_metric_name(run_dir: Path, dataset_name: str) -> str:
    result_json = load_results_json(run_dir, dataset_name)
    for metric_name in METRIC_CANDIDATES:
        if f"test_{dataset_name}_{metric_name}" in result_json:
            return metric_name
    raise ValueError(f"Could not infer a metric for dataset {dataset_name}.")


def dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name.split("/")[-1])


def parse_datasets_from_script(script_text: str) -> list[str]:
    match = re.search(r"--datasets\s+(.+?)\s+--split", script_text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Could not parse datasets from the eval script.")
    return match.group(1).split()


def mean_context_length(run_dir: Path, dataset_name: str) -> float:
    path = run_dir / f"test_{dataset_name}_generated_text.jsonl"
    total = 0.0
    count = 0
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            record = json.loads(line)
            ctx_ids_len = record.get("ctx_ids_len")
            if isinstance(ctx_ids_len, (int, float)):
                total += float(ctx_ids_len)
                count += 1
    if count == 0:
        raise ValueError(f"Could not find ctx_ids_len values in {path}.")
    return total / count


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()

    script_text = load_script_text(args.eval_script)
    checkpoint_path = parse_checkpoint_from_script(script_text)
    eval_results_root = parse_eval_results_root(checkpoint_path)
    expected_datasets = parse_datasets_from_script(script_text)
    selected_runs = choose_latest_runs_by_chunk(eval_results_root, expected_datasets)

    if not selected_runs:
        raise SystemExit(f"No matching eval runs found under {eval_results_root}.")

    output_dir = (
        eval_results_root / "plots" if args.output_dir is None else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    for dataset_name in expected_datasets:
        metric_name = infer_metric_name(selected_runs[0].run_dir, dataset_name)
        reference_performance = REFERENCE_PERFORMANCE.get(dataset_name)
        chunk_sizes: list[int] = []
        approx_num_chunks: list[float] = []
        normalized_performance_values: list[float] = []

        for run_info in selected_runs:
            chunk_size = run_info.chunk_size
            if chunk_size == 65536:
                continue
            performance = get_performance_value(
                run_info.run_dir,
                dataset_name,
                metric_name=metric_name,
            )
            if performance is None:
                continue

            avg_context_length = mean_context_length(run_info.run_dir, dataset_name)
            chunk_sizes.append(chunk_size)
            approx_num_chunks.append(avg_context_length / chunk_size)
            plot_value = (
                performance / reference_performance
                if reference_performance not in (None, 0)
                else performance
            )
            normalized_performance_values.append(plot_value)

        if not chunk_sizes:
            continue

        fig, ax_perf = plt.subplots(
            1,
            1,
            figsize=(7.5, 5.5),
            constrained_layout=False,
        )
        fig.subplots_adjust(bottom=0.18, top=0.86)

        ax_perf.plot(
            chunk_sizes,
            normalized_performance_values,
            marker="o",
            linewidth=2,
            color="#1f77b4",
            markeredgecolor="black",
            markeredgewidth=1,
        )
        for chunk_size, performance, approx_chunks in zip(
            chunk_sizes,
            normalized_performance_values,
            approx_num_chunks,
        ):
            ax_perf.annotate(
                approx_chunk_label(approx_chunks),
                xy=(chunk_size, performance),
                xytext=(0, 7),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=11,
                color="#4c4f69",
            )

        ax_perf.grid(True, alpha=0.25)
        ax_perf.set_xscale("log", base=2)
        ax_perf.set_xticks(chunk_sizes)
        ax_perf.set_xticklabels(
            [chunk_size_tick_label(chunk_size) for chunk_size in chunk_sizes]
        )
        ax_perf.xaxis.set_minor_locator(plt.NullLocator())

        pretty_name = dataset_label(dataset_name)
        ax_perf.set_title(f"Retrieval Performance\n({pretty_name})", fontweight="bold")
        ax_perf.set_xlabel("Chunk Size (tokens)")
        ax_perf.set_ylabel(
            "Normalized Performance"
            if reference_performance not in (None, 0)
            else "Performance"
        )
        ax_perf.set_ylim(bottom=0.0)

        output_path = ensure_output_path(
            output_dir / f"d2l-chunking-abalation-{dataset_name.split('/')[-1]}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
