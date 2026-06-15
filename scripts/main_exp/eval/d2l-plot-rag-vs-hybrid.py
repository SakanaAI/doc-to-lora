from __future__ import annotations

import argparse
import csv
import json
import re
import shlex
from collections import defaultdict, deque
from dataclasses import dataclass
from pathlib import Path

import matplotlib
import matplotlib as mpl

matplotlib.use("Agg")
import matplotlib.pyplot as plt

LATTE_STYLE = "https://raw.githubusercontent.com/51616/catppuccin-matplotlib/main/src/mplcatppuccin/data/latte.mplstyle"
BASE_MODEL_EVAL_ROOT = Path("eval_results/google/gemma-2-2b-it")
DATASET_LABELS = {
    "oolong-synth": "Oolong-Synth",
    "longbench/qasper_e": "Qasper",
    "longbench/2wikimqa_e": "2WikiMQA",
    "longbench/multifieldqa_en_e": "MultiFieldQA-EN",
    "longbench/gov_report_e": "GovReport",
}
METRIC_CANDIDATES = (
    "oolong_score",
    "qa_f1_score",
    "rougeL.f1",
    "qa_f1",
    "f1",
    "accuracy",
)
METRIC_LABELS = {
    "oolong_score": "Oolong Score",
    "qa_f1_score": "QA F1 Score",
    "rougeL.f1": "ROUGE-L F1",
    "qa_f1": "QA F1",
    "f1": "F1",
    "accuracy": "Accuracy",
}
COMPARISON_TOLERANCE = 0.2
CMD_RE = re.compile(r"CMD:\s+(.*)$", re.MULTILINE)
BASELINE_EXCLUDE_FLAGS = {
    "--use_rag",
    "--use_hybrid_rag",
    "--use_cd",
    "--use_llmlingua",
    "--use_t2l",
}
NO_CONTEXT_COLUMN = "No context"
CSV_SUBCOLUMNS = (NO_CONTEXT_COLUMN, "top-1", "top-4")


@dataclass(frozen=True)
class RagRunInfo:
    run_dir: Path
    run_name: str
    datasets: tuple[str, ...]
    is_hybrid: bool
    top_k: int
    chunk_size: int
    chunk_overlap: int
    max_retrieved_tokens: int


def configure_plot_style() -> None:
    plt.style.use(["ggplot", LATTE_STYLE])
    plt.rcParams["axes.facecolor"] = "F7FBFC"
    plt.rcParams["figure.facecolor"] = "white"
    plt.rcParams["savefig.facecolor"] = "white"
    plt.rcParams["grid.color"] = "cccccc"
    plt.rcParams["grid.linewidth"] = 1
    plt.rcParams["axes.edgecolor"] = "ccd0da"
    plt.rcParams["legend.facecolor"] = "white"
    plt.rcParams["legend.fontsize"] = 12
    plt.rcParams["xtick.labelsize"] = 13
    plt.rcParams["ytick.labelsize"] = 13
    plt.rcParams["axes.labelweight"] = "bold"
    plt.rcParams["axes.prop_cycle"] = mpl.cycler(color=plt.cm.tab10.colors)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Plot one raw-RAG vs hybrid-RAG comparison figure per dataset.",
    )
    parser.add_argument(
        "--rag-root",
        type=Path,
        default=Path("eval_results/google/gemma-2-2b-it"),
        help="Directory containing raw RAG eval runs.",
    )
    parser.add_argument(
        "--hybrid-root",
        type=Path,
        default=Path("trained_d2l/gemma_2b_d2l/eval-results-20000"),
        help="Directory containing hybrid RAG eval runs.",
    )
    parser.add_argument(
        "--rag-chunk-size",
        type=int,
        default=256,
        help="Filter runs by rag chunk size.",
    )
    parser.add_argument(
        "--rag-chunk-overlap",
        type=int,
        default=64,
        help="Filter runs by rag chunk overlap.",
    )
    parser.add_argument(
        "--rag-max-retrieved-tokens",
        type=int,
        default=1536,
        help="Filter runs by rag max retrieved tokens.",
    )
    parser.add_argument(
        "--datasets",
        nargs="*",
        default=None,
        help="Optional dataset allowlist. Defaults to all datasets with matched raw/hybrid runs.",
    )
    parser.add_argument(
        "--top-ks",
        nargs="*",
        type=int,
        default=None,
        help="Optional top-k allowlist. Defaults to all matched top-k values.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Optional output directory. Defaults to HYBRID_ROOT/plots.",
    )
    return parser


def _extract_flag_value(tokens: list[str], flag: str) -> str | None:
    for index, token in enumerate(tokens):
        if token == flag and index + 1 < len(tokens):
            return tokens[index + 1]
        if token.startswith(f"{flag}="):
            return token.split("=", 1)[1]
    return None


def _extract_datasets(tokens: list[str]) -> tuple[str, ...]:
    if "--datasets" not in tokens:
        return ()

    index = tokens.index("--datasets") + 1
    datasets: list[str] = []
    while index < len(tokens) and not tokens[index].startswith("--"):
        datasets.append(tokens[index])
        index += 1
    return tuple(datasets)


def _has_max_test_samples_flag(tokens: list[str]) -> bool:
    return _extract_flag_value(tokens, "--max_test_samples_per_ds") is not None


def parse_rag_run_info(run_dir: Path) -> RagRunInfo | None:
    log_path = run_dir / "debug.log"
    if not log_path.exists():
        return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = CMD_RE.search(log_text)
    if match is None:
        return None

    command = match.group(1).strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    use_rag = "--use_rag" in tokens
    use_hybrid_rag = "--use_hybrid_rag" in tokens
    if use_rag == use_hybrid_rag:
        return None
    if _has_max_test_samples_flag(tokens):
        return None

    datasets = _extract_datasets(tokens)
    if not datasets:
        return None

    top_k_value = _extract_flag_value(tokens, "--rag_top_k")
    chunk_size_value = _extract_flag_value(tokens, "--rag_chunk_size")
    chunk_overlap_value = _extract_flag_value(tokens, "--rag_chunk_overlap")
    max_tokens_value = _extract_flag_value(tokens, "--rag_max_retrieved_tokens")
    if None in (
        top_k_value,
        chunk_size_value,
        chunk_overlap_value,
        max_tokens_value,
    ):
        return None

    return RagRunInfo(
        run_dir=run_dir,
        run_name=run_dir.name,
        datasets=datasets,
        is_hybrid=use_hybrid_rag,
        top_k=int(top_k_value),
        chunk_size=int(chunk_size_value),
        chunk_overlap=int(chunk_overlap_value),
        max_retrieved_tokens=int(max_tokens_value),
    )


def parse_plain_base_run_datasets(run_dir: Path) -> tuple[str, ...] | None:
    log_path = run_dir / "debug.log"
    if not log_path.exists():
        return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = CMD_RE.search(log_text)
    if match is None:
        return None

    command = match.group(1).strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if "--model_name_or_path" not in tokens:
        return None
    if any(flag in tokens for flag in BASELINE_EXCLUDE_FLAGS):
        return None
    if _has_max_test_samples_flag(tokens):
        return None

    datasets = _extract_datasets(tokens)
    if not datasets:
        return None
    return datasets


def parse_plain_d2l_run_datasets(run_dir: Path) -> tuple[str, ...] | None:
    log_path = run_dir / "debug.log"
    if not log_path.exists():
        return None

    log_text = log_path.read_text(encoding="utf-8", errors="replace")
    match = CMD_RE.search(log_text)
    if match is None:
        return None

    command = match.group(1).strip()
    try:
        tokens = shlex.split(command)
    except ValueError:
        tokens = command.split()

    if "--checkpoint_path" not in tokens:
        return None
    if any(flag in tokens for flag in BASELINE_EXCLUDE_FLAGS):
        return None
    if "--add_ctx_to_input" in tokens:
        return None
    if _has_max_test_samples_flag(tokens):
        return None
    max_ctx_chunk_len = _extract_flag_value(tokens, "--max_ctx_chunk_len")
    if max_ctx_chunk_len != "8192":
        return None

    datasets = _extract_datasets(tokens)
    if not datasets:
        return None
    return datasets


def result_json_path(run_dir: Path, dataset_name: str) -> Path:
    return run_dir / f"test_{dataset_name}_results.json"


def load_results_json(run_dir: Path, dataset_name: str) -> dict[str, object]:
    return json.loads(
        result_json_path(run_dir, dataset_name).read_text(encoding="utf-8")
    )


def infer_metric_name(result_json: dict[str, object], dataset_name: str) -> str:
    for metric_name in METRIC_CANDIDATES:
        if f"test_{dataset_name}_{metric_name}" in result_json:
            return metric_name
    raise ValueError(f"Could not infer a metric for dataset {dataset_name}.")


def metric_label(metric_name: str) -> str:
    return METRIC_LABELS.get(metric_name, metric_name.replace("_", " ").title())


def dataset_label(dataset_name: str) -> str:
    return DATASET_LABELS.get(dataset_name, dataset_name.split("/")[-1])


def select_latest_runs(
    root: Path,
    *,
    want_hybrid: bool,
    allowed_datasets: set[str] | None,
    allowed_top_ks: set[int] | None,
    rag_chunk_size: int,
    rag_chunk_overlap: int,
    rag_max_retrieved_tokens: int,
) -> dict[tuple[str, int], RagRunInfo]:
    selected: dict[tuple[str, int], RagRunInfo] = {}
    if not root.exists():
        return selected

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info = parse_rag_run_info(run_dir)
        if run_info is None or run_info.is_hybrid != want_hybrid:
            continue
        if run_info.chunk_size != rag_chunk_size:
            continue
        if run_info.chunk_overlap != rag_chunk_overlap:
            continue
        if run_info.max_retrieved_tokens != rag_max_retrieved_tokens:
            continue
        if allowed_top_ks is not None and run_info.top_k not in allowed_top_ks:
            continue

        for dataset_name in run_info.datasets:
            if allowed_datasets is not None and dataset_name not in allowed_datasets:
                continue
            if not result_json_path(run_info.run_dir, dataset_name).exists():
                continue

            key = (dataset_name, run_info.top_k)
            current = selected.get(key)
            if current is None or run_info.run_name > current.run_name:
                selected[key] = run_info

    return selected


def get_metric_value(
    run_dir: Path, dataset_name: str, metric_name: str
) -> float | None:
    result_json = load_results_json(run_dir, dataset_name)
    value = result_json.get(f"test_{dataset_name}_{metric_name}")
    if value in (None, "None", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def select_latest_plain_base_runs(
    root: Path,
    datasets: set[str],
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    if not root.exists():
        return selected

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_datasets = parse_plain_base_run_datasets(run_dir)
        if run_datasets is None:
            continue

        for dataset_name in run_datasets:
            if dataset_name not in datasets:
                continue
            if not result_json_path(run_dir, dataset_name).exists():
                continue
            selected[dataset_name] = run_dir

    return selected


def select_latest_plain_d2l_runs(
    root: Path,
    datasets: set[str],
) -> dict[str, Path]:
    selected: dict[str, Path] = {}
    if not root.exists():
        return selected

    for run_dir in sorted(root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_datasets = parse_plain_d2l_run_datasets(run_dir)
        if run_datasets is None:
            continue

        for dataset_name in run_datasets:
            if dataset_name not in datasets:
                continue
            if not result_json_path(run_dir, dataset_name).exists():
                continue
            selected[dataset_name] = run_dir

    return selected


CSV_TOP_KS = (1, 4)


def csv_output_path(output_dir: Path) -> Path:
    return output_dir / "d2l-rag-vs-hybrid.csv"


def confusion_output_path_for_dataset(output_dir: Path, dataset_name: str) -> Path:
    return output_dir / f"d2l-rag-vs-hybrid-confusion-{dataset_name.split('/')[-1]}.png"


def generated_text_path(run_dir: Path, dataset_name: str) -> Path:
    return run_dir / f"test_{dataset_name}_generated_text.jsonl"


def sample_key(sample: dict[str, object], dataset_name: str) -> str:
    if dataset_name == "oolong-synth":
        return "oolong:" + str(sample.get("oolong_id"))

    input_text = str(sample.get("input", ""))
    label_text = str(sample.get("label", ""))
    ctx_ids_len = str(sample.get("ctx_ids_len", ""))
    return f"{dataset_name}\n{input_text}\n{label_text}\n{ctx_ids_len}"


def sample_metric_value(sample: dict[str, object], metric_name: str) -> float:
    value = sample.get(metric_name)
    if not isinstance(value, (int, float)):
        return 0.0
    return float(value)


def load_metric_by_sample(
    run_dir: Path,
    dataset_name: str,
    metric_name: str,
) -> dict[str, deque[float]]:
    path = generated_text_path(run_dir, dataset_name)
    grouped: dict[str, deque[float]] = defaultdict(deque)
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            sample = json.loads(line)
            grouped[sample_key(sample, dataset_name)].append(
                sample_metric_value(sample, metric_name)
            )
    return grouped


def relative_bucket(score: float, other_score: float) -> int:
    delta = score - other_score
    if delta > COMPARISON_TOLERANCE:
        return 2
    if delta < -COMPARISON_TOLERANCE:
        return 0
    return 1


def build_confusion_counts(
    raw_run_dir: Path,
    hybrid_run_dir: Path,
    dataset_name: str,
    metric_name: str,
) -> list[list[int]]:
    raw_grouped = load_metric_by_sample(raw_run_dir, dataset_name, metric_name)
    hybrid_grouped = load_metric_by_sample(hybrid_run_dir, dataset_name, metric_name)
    raw_counts = {key: len(values) for key, values in raw_grouped.items()}
    hybrid_counts = {key: len(values) for key, values in hybrid_grouped.items()}
    if raw_counts != hybrid_counts:
        raise ValueError(f"Could not align raw and hybrid samples for {dataset_name}.")

    matrix = [[0, 0, 0], [0, 0, 0], [0, 0, 0]]
    for key in sorted(raw_grouped):
        raw_values = raw_grouped[key]
        hybrid_values = hybrid_grouped[key]
        while raw_values:
            raw_score = raw_values.popleft()
            hybrid_score = hybrid_values.popleft()
            raw_bucket = relative_bucket(raw_score, hybrid_score)
            hybrid_bucket = relative_bucket(hybrid_score, raw_score)
            matrix[raw_bucket][hybrid_bucket] += 1
    return matrix


def plot_confusion_matrices(
    output_dir: Path,
    dataset_name: str,
    metric_name: str,
    paired_points: list[tuple[int, float, float]],
    raw_runs: dict[tuple[str, int], RagRunInfo],
    hybrid_runs: dict[tuple[str, int], RagRunInfo],
) -> None:
    ncols = len(paired_points)
    fig, axes = plt.subplots(
        1,
        ncols,
        figsize=(4.8 * ncols, 4.6),
        constrained_layout=False,
    )
    if ncols == 1:
        axes = [axes]
    fig.subplots_adjust(bottom=0.16, top=0.80, wspace=0.35)

    for ax, (top_k, _, _) in zip(axes, paired_points):
        matrix = build_confusion_counts(
            raw_runs[(dataset_name, top_k)].run_dir,
            hybrid_runs[(dataset_name, top_k)].run_dir,
            dataset_name,
            metric_name,
        )
        image = ax.imshow(matrix, cmap="Blues")
        total = sum(sum(row) for row in matrix)
        max_cell = max(max(row) for row in matrix) if total else 0

        for raw_index in range(3):
            for hybrid_index in range(3):
                count = matrix[raw_index][hybrid_index]
                pct = 100.0 * count / total if total else 0.0
                color = "white" if count > max_cell / 2 else "#4c4f69"
                ax.text(
                    hybrid_index,
                    raw_index,
                    f"{count}\n({pct:.1f}%)",
                    ha="center",
                    va="center",
                    color=color,
                    fontsize=11,
                    fontweight="bold" if count == max_cell and count > 0 else None,
                )

        ax.set_xticks([0, 1, 2])
        ax.set_yticks([0, 1, 2])
        ax.set_xticklabels(["More\nIncorrect", "Same", "More\nCorrect"])
        ax.set_yticklabels(["More\nIncorrect", "Same", "More\nCorrect"])
        ax.set_xlabel("Hybrid RAG")
        if ax is axes[0]:
            ax.set_ylabel("Raw RAG")
        ax.set_title(f"top-k = {top_k}", fontweight="bold")
        ax.grid(False)

        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_color("#ccd0da")

        fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)

    fig.suptitle(
        f"Relative Score Agreement\n({dataset_label(dataset_name)}, delta threshold = {COMPARISON_TOLERANCE:.1f})",
        fontweight="bold",
        fontsize=16,
    )
    output_path = confusion_output_path_for_dataset(output_dir, dataset_name)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    print(f"Saved plot to {output_path}")
    plt.close(fig)


def write_summary_csv(
    output_dir: Path,
    datasets: list[str],
    summary: dict[str, dict[str, dict[str, float | str | None]]],
) -> None:
    output_path = csv_output_path(output_dir)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        top_header = [""]
        sub_header = ["Method"]
        for dataset_name in datasets:
            pretty_name = dataset_label(dataset_name)
            top_header.extend([pretty_name] * len(CSV_SUBCOLUMNS))
            sub_header.extend(CSV_SUBCOLUMNS)

        writer.writerow(top_header)
        writer.writerow(sub_header)

        for method_name in ("RAG", "RAG + D2L", "D2L"):
            row = [method_name]
            for dataset_name in datasets:
                method_values = summary.get(method_name, {}).get(dataset_name, {})
                for subcolumn in CSV_SUBCOLUMNS:
                    value = method_values.get(subcolumn)
                    if isinstance(value, (int, float)):
                        row.append(f"{float(value):.4f}")
                    elif value is None:
                        row.append("")
                    else:
                        row.append(str(value))
            writer.writerow(row)

    print(f"Saved CSV to {output_path}")


def main() -> None:
    args = build_parser().parse_args()
    configure_plot_style()

    allowed_datasets = set(args.datasets) if args.datasets else None
    allowed_top_ks = set(args.top_ks) if args.top_ks else None
    output_dir = (
        args.hybrid_root / "plots" if args.output_dir is None else args.output_dir
    )
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_runs = select_latest_runs(
        args.rag_root,
        want_hybrid=False,
        allowed_datasets=allowed_datasets,
        allowed_top_ks=allowed_top_ks,
        rag_chunk_size=args.rag_chunk_size,
        rag_chunk_overlap=args.rag_chunk_overlap,
        rag_max_retrieved_tokens=args.rag_max_retrieved_tokens,
    )
    hybrid_runs = select_latest_runs(
        args.hybrid_root,
        want_hybrid=True,
        allowed_datasets=allowed_datasets,
        allowed_top_ks=allowed_top_ks,
        rag_chunk_size=args.rag_chunk_size,
        rag_chunk_overlap=args.rag_chunk_overlap,
        rag_max_retrieved_tokens=args.rag_max_retrieved_tokens,
    )

    shared_keys = set(raw_runs) & set(hybrid_runs)
    if not shared_keys:
        raise SystemExit("Could not find any matched raw-RAG and hybrid-RAG runs.")

    datasets = sorted({dataset_name for dataset_name, _ in shared_keys})
    base_runs = select_latest_plain_base_runs(BASE_MODEL_EVAL_ROOT, set(datasets))
    d2l_runs = select_latest_plain_d2l_runs(args.hybrid_root, set(datasets))
    summary: dict[str, dict[str, dict[str, float | str | None]]] = {
        "RAG": {},
        "RAG + D2L": {},
        "D2L": {},
    }
    for dataset_name in datasets:
        base_run_dir = base_runs.get(dataset_name)
        if base_run_dir is None:
            continue

        shared_top_ks = sorted(
            top_k
            for candidate_dataset, top_k in shared_keys
            if candidate_dataset == dataset_name
        )
        if not shared_top_ks:
            continue

        raw_result_json = load_results_json(
            raw_runs[(dataset_name, shared_top_ks[0])].run_dir, dataset_name
        )
        hybrid_result_json = load_results_json(
            hybrid_runs[(dataset_name, shared_top_ks[0])].run_dir,
            dataset_name,
        )
        metric_name = infer_metric_name(raw_result_json, dataset_name)
        hybrid_metric_name = infer_metric_name(hybrid_result_json, dataset_name)
        if metric_name != hybrid_metric_name:
            raise ValueError(
                f"Metric mismatch for {dataset_name}: raw uses {metric_name}, hybrid uses {hybrid_metric_name}."
            )
        base_result_json = load_results_json(base_run_dir, dataset_name)
        base_metric_name = infer_metric_name(base_result_json, dataset_name)
        if metric_name != base_metric_name:
            raise ValueError(
                f"Metric mismatch for {dataset_name}: comparison uses {metric_name}, baseline uses {base_metric_name}."
            )
        base_value = get_metric_value(base_run_dir, dataset_name, metric_name)
        if base_value in (None, 0):
            continue

        d2l_run_dir = d2l_runs.get(dataset_name)
        d2l_value = None
        if d2l_run_dir is not None:
            d2l_result_json = load_results_json(d2l_run_dir, dataset_name)
            d2l_metric_name = infer_metric_name(d2l_result_json, dataset_name)
            if d2l_metric_name == metric_name:
                candidate_value = get_metric_value(
                    d2l_run_dir, dataset_name, metric_name
                )
                if candidate_value is not None:
                    d2l_value = candidate_value / base_value

        paired_points: list[tuple[int, float, float]] = []
        for top_k in shared_top_ks:
            raw_value = get_metric_value(
                raw_runs[(dataset_name, top_k)].run_dir, dataset_name, metric_name
            )
            hybrid_value = get_metric_value(
                hybrid_runs[(dataset_name, top_k)].run_dir,
                dataset_name,
                metric_name,
            )
            if raw_value is None or hybrid_value is None:
                continue
            paired_points.append(
                (top_k, raw_value / base_value, hybrid_value / base_value)
            )

        if not paired_points:
            continue

        summary["RAG"][dataset_name] = {
            NO_CONTEXT_COLUMN: "N/A",
            **{f"top-{top_k}": raw_value for top_k, raw_value, _ in paired_points},
        }
        summary["RAG + D2L"][dataset_name] = {
            NO_CONTEXT_COLUMN: "N/A",
            **{
                f"top-{top_k}": hybrid_value for top_k, _, hybrid_value in paired_points
            },
        }
        summary["D2L"][dataset_name] = {
            NO_CONTEXT_COLUMN: d2l_value,
            "top-1": "N/A",
            "top-4": "N/A",
        }
        plot_confusion_matrices(
            output_dir,
            dataset_name,
            metric_name,
            paired_points,
            raw_runs,
            hybrid_runs,
        )

    write_summary_csv(output_dir, datasets, summary)


if __name__ == "__main__":
    main()
