from __future__ import annotations

import argparse
import math
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
BASE_MODEL_EVAL_ROOT = Path("eval_results/google/gemma-2-2b-it")
METRIC_CANDIDATES = (
    "qa_f1_score",
    "rougeL.f1",
    "qa_f1",
    "f1",
    "accuracy",
)
LEN_KEY_RE = re.compile(r"^test_(.+)_(.+)_len_([0-9]+)-([0-9]+|inf)$")
MIN_CONTEXT_LENGTH = 4096
MAX_CONTEXT_LENGTH = 16384


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
        description="Plot LongBench performance vs context length for the 8192 chunk-size D2L eval run.",
    )
    parser.add_argument(
        "--eval-script",
        type=Path,
        default=Path(__file__).with_name("d2l-chunking-abalation.sh"),
        help="Path to the eval sweep shell script.",
    )
    parser.add_argument(
        "--chunk-size",
        type=int,
        default=8192,
        help="Chunk size to select from the sweep.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to eval-results-*/plots/.",
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


def parse_datasets_from_script(script_text: str) -> list[str]:
    match = re.search(r"--datasets\s+(.+?)\s+--split", script_text, flags=re.DOTALL)
    if match is None:
        raise ValueError("Could not parse datasets from the eval script.")
    return match.group(1).split()


def infer_metric_name(result_json: dict[str, object], dataset_name: str) -> str:
    for metric_name in METRIC_CANDIDATES:
        if f"test_{dataset_name}_{metric_name}" in result_json:
            return metric_name
    raise ValueError(f"Could not infer a metric for dataset {dataset_name}.")


def dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name.split("/")[-1])


def choose_latest_complete_base_run(expected_datasets: list[str]) -> Path:
    candidates: list[Path] = []
    for run_dir in sorted(BASE_MODEL_EVAL_ROOT.iterdir()):
        if not run_dir.is_dir():
            continue
        if all(
            (run_dir / f"test_{dataset_name}_results.json").exists()
            for dataset_name in expected_datasets
        ):
            candidates.append(run_dir)

    if not candidates:
        raise ValueError(
            f"Could not find a complete base-model run under {BASE_MODEL_EVAL_ROOT}."
        )
    return candidates[-1]


def extract_length_series(
    result_json: dict[str, object],
    dataset_name: str,
    metric_name: str,
) -> tuple[list[int], list[float]]:
    xs: list[int] = []
    ys: list[float] = []
    metric_prefix = f"test_{dataset_name}_{metric_name}_len_"
    count_prefix = f"test_{dataset_name}_num_samples_{metric_name}_len_"

    for key, value in result_json.items():
        if not key.startswith(metric_prefix):
            continue
        match = LEN_KEY_RE.match(key)
        if match is None:
            continue

        low = int(match.group(3))
        high_str = match.group(4)
        if low == 0 or high_str == "inf":
            continue
        if value in (None, "None", "N/A"):
            continue

        high = int(high_str)
        upper_bound = high + 1
        if upper_bound < MIN_CONTEXT_LENGTH or upper_bound > MAX_CONTEXT_LENGTH:
            continue
        count_key = f"{count_prefix}{low}-{high_str}"
        count = result_json.get(count_key)
        if not isinstance(count, int) or count <= 0:
            continue

        xs.append(upper_bound)
        ys.append(float(value))

    pairs = sorted(zip(xs, ys), key=lambda pair: pair[0])
    return [x for x, _ in pairs], [y for _, y in pairs]


def tick_label(length_value: int) -> str:
    exponent = int(round(math.log2(length_value)))
    return rf"$2^{{{exponent}}}$"


def to_series_map(xs: list[int], ys: list[float]) -> dict[int, float]:
    return {x: y for x, y in zip(xs, ys)}


def chunk_count_label(context_length: int, chunk_size: int) -> str:
    num_chunks = max(1, math.ceil(context_length / chunk_size))
    suffix = "" if num_chunks == 1 else "s"
    return f"{num_chunks} chunk{suffix}"


def chunk_label_offset(dataset_name: str) -> tuple[int, int]:
    if dataset_name == "longbench/2wikimqa_e":
        return (0, -15)
    return (0, 11)


def chunk_label_position(
    dataset_name: str,
    index: int,
    total: int,
) -> tuple[tuple[int, int], str]:
    dx, dy = chunk_label_offset(dataset_name)
    ha = "center"
    if total > 1 and index == 0:
        dx += 2
    elif total > 1 and index == total - 1:
        dx -= 2
    return (dx, dy), ha


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()

    script_text = load_script_text(args.eval_script)
    checkpoint_path = parse_checkpoint_from_script(script_text)
    eval_results_root = parse_eval_results_root(checkpoint_path)
    expected_datasets = parse_datasets_from_script(script_text)
    selected_runs = choose_latest_runs_by_chunk(eval_results_root, expected_datasets)
    base_run_dir = choose_latest_complete_base_run(expected_datasets)

    target_run = next(
        (
            run_info
            for run_info in selected_runs
            if run_info.chunk_size == args.chunk_size
        ),
        None,
    )
    if target_run is None:
        raise SystemExit(
            f"Could not find a run with chunk size {args.chunk_size} under {eval_results_root}."
        )

    output_dir = eval_results_root / "plots" if args.output is None else args.output
    output_dir.mkdir(parents=True, exist_ok=True)

    color_map = plt.get_cmap("tab10", 2)
    for dataset_name in expected_datasets:
        result_json = load_results_json(target_run.run_dir, dataset_name)
        metric_name = infer_metric_name(result_json, dataset_name)
        xs, ys = extract_length_series(result_json, dataset_name, metric_name)
        base_result_json = load_results_json(base_run_dir, dataset_name)
        base_xs, base_ys = extract_length_series(
            base_result_json, dataset_name, metric_name
        )

        d2l_series = to_series_map(xs, ys)
        base_series = to_series_map(base_xs, base_ys)
        normalized_xs = [
            x for x in sorted(set(d2l_series) & set(base_series)) if base_series[x] != 0
        ]
        all_xs = sorted(set(xs) | set(base_xs))
        if not all_xs:
            continue

        fig, (ax_raw, ax_norm) = plt.subplots(
            1,
            2,
            figsize=(14.5, 5.8),
            constrained_layout=False,
        )
        fig.subplots_adjust(bottom=0.16, top=0.84, wspace=0.28)

        if xs:
            ax_raw.plot(
                xs,
                ys,
                marker="o",
                linewidth=2,
                label="D2L",
                markeredgecolor="black",
                markeredgewidth=1,
                color=color_map(0),
            )
            for i, (x, y) in enumerate(zip(xs, ys)):
                raw_label_offset, raw_ha = chunk_label_position(
                    dataset_name, i, len(xs)
                )
                ax_raw.annotate(
                    chunk_count_label(x, args.chunk_size),
                    (x, y),
                    textcoords="offset points",
                    xytext=raw_label_offset,
                    ha=raw_ha,
                    fontsize=8,
                    color="#6c6f85",
                )
        if base_xs:
            ax_raw.plot(
                base_xs,
                base_ys,
                linestyle="--",
                marker="o",
                linewidth=2,
                label="Base",
                color=color_map(1),
                alpha=0.9,
                markeredgecolor="black",
                markeredgewidth=1,
            )

        ax_raw.set_xscale("log", base=2)
        ax_raw.set_xticks(all_xs)
        ax_raw.set_xticklabels([tick_label(x) for x in all_xs])
        ax_raw.xaxis.set_minor_locator(plt.NullLocator())
        ax_raw.grid(True, alpha=0.25)
        ax_raw.set_title("Raw Performance", fontweight="bold")
        ax_raw.set_xlabel("Context Length (tokens)")
        ax_raw.set_ylabel("QA F1 Score")
        ax_raw.set_ylim(-0.05, 1.05)
        ax_raw.legend(loc="lower left", frameon=True, fancybox=True)

        if normalized_xs:
            normalized_d2l_ys = [d2l_series[x] / base_series[x] for x in normalized_xs]
            normalized_base_ys = [1.0 for _ in normalized_xs]
            ax_norm.plot(
                normalized_xs,
                normalized_d2l_ys,
                marker="o",
                linewidth=2,
                label="D2L",
                markeredgecolor="black",
                markeredgewidth=1,
                color=color_map(0),
            )
            for i, (x, y) in enumerate(zip(normalized_xs, normalized_d2l_ys)):
                norm_label_offset, norm_ha = chunk_label_position(
                    dataset_name,
                    i,
                    len(normalized_xs),
                )
                ax_norm.annotate(
                    chunk_count_label(x, args.chunk_size),
                    (x, y),
                    textcoords="offset points",
                    xytext=norm_label_offset,
                    ha=norm_ha,
                    fontsize=8,
                    color="#6c6f85",
                )
            ax_norm.plot(
                normalized_xs,
                normalized_base_ys,
                linestyle="--",
                marker="o",
                linewidth=2,
                label="Base",
                color=color_map(1),
                alpha=0.9,
                markeredgecolor="black",
                markeredgewidth=1,
            )
        else:
            ax_norm.text(
                0.5,
                0.5,
                "No overlapping bins for normalization.",
                ha="center",
                va="center",
                transform=ax_norm.transAxes,
            )

        norm_ticks = normalized_xs if normalized_xs else all_xs
        ax_norm.set_xscale("log", base=2)
        ax_norm.set_xticks(norm_ticks)
        ax_norm.set_xticklabels([tick_label(x) for x in norm_ticks])
        ax_norm.xaxis.set_minor_locator(plt.NullLocator())
        ax_norm.grid(True, alpha=0.25)
        ax_norm.set_title("Normalized Performance", fontweight="bold")
        ax_norm.set_xlabel("Context Length (tokens)")
        ax_norm.set_ylabel("Normalized Performance")
        ax_norm.set_ylim(bottom=0.0)
        if normalized_xs:
            ax_norm.legend(loc="lower left", frameon=True, fancybox=True)

        fig.suptitle(
            f"Performance Grouped by Length\n({dataset_label(dataset_name)}, chunk size = {args.chunk_size})",
            fontweight="bold",
            fontsize=16,
        )

        output_path = (
            output_dir
            / f"d2l-context-length-{args.chunk_size}-{dataset_name.split('/')[-1]}.png"
        )
        fig.savefig(output_path, dpi=200, bbox_inches="tight")
        print(f"Saved plot to {output_path}")
        plt.close(fig)


if __name__ == "__main__":
    main()
