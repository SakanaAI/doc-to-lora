from __future__ import annotations

import json
import re
import shlex
from dataclasses import dataclass
from pathlib import Path

DATASET_RE = re.compile(r"ctx_magic_number_(\d+)_(\d+)")
CHECKPOINT_RE = re.compile(r"^CHECKPOINT_PATH=(.+)$", re.MULTILINE)
CMD_RE = re.compile(r"CMD:\s+(.*)$", re.MULTILINE)


@dataclass(frozen=True)
class RunInfo:
    run_dir: Path
    run_name: str
    chunk_size: int
    ctx_chunk_overlap: int
    datasets: tuple[str, ...]
    has_dense_logging: bool


def load_script_text(script_path: Path) -> str:
    return script_path.read_text(encoding="utf-8")


def parse_checkpoint_path(script_text: str) -> Path:
    match = CHECKPOINT_RE.search(script_text)
    if match is None:
        raise ValueError("Could not find CHECKPOINT_PATH in the eval script.")
    return Path(match.group(1).strip().strip("\"'"))


def parse_expected_datasets(script_text: str) -> list[str]:
    datasets = {match.group(0) for match in DATASET_RE.finditer(script_text)}
    if not datasets:
        raise ValueError(
            "Could not find any ctx_magic_number datasets in the eval script."
        )
    return sorted(datasets, key=dataset_context_length)


def parse_eval_results_root(checkpoint_path: Path) -> Path:
    checkpoint_match = re.search(r"checkpoint-(\d+)", checkpoint_path.as_posix())
    if checkpoint_match is None:
        raise ValueError(f"Could not parse checkpoint step from {checkpoint_path}.")
    checkpoint_step = checkpoint_match.group(1)
    return checkpoint_path.parent.parent / f"eval-results-{checkpoint_step}"


def parse_dataset_context_bounds(dataset_name: str) -> tuple[int, int]:
    match = DATASET_RE.fullmatch(dataset_name)
    if match is None:
        raise ValueError(f"Unsupported dataset name: {dataset_name}")
    return int(match.group(1)), int(match.group(2))


def dataset_context_length(dataset_name: str) -> int:
    return parse_dataset_context_bounds(dataset_name)[1]


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


def parse_run_info(run_dir: Path) -> RunInfo | None:
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

    chunk_size_value = _extract_flag_value(tokens, "--max_ctx_chunk_len")
    if chunk_size_value is None:
        return None

    datasets = _extract_datasets(tokens)
    if not datasets:
        return None

    return RunInfo(
        run_dir=run_dir,
        run_name=run_dir.name,
        chunk_size=int(chunk_size_value),
        ctx_chunk_overlap=int(_extract_flag_value(tokens, "--ctx_chunk_overlap") or 0),
        datasets=datasets,
        has_dense_logging="--log_dense_lora_magnitude" in tokens,
    )


def _run_has_expected_outputs(run_dir: Path, datasets: list[str]) -> bool:
    return all(
        (run_dir / f"test_{dataset_name}_results.json").exists()
        for dataset_name in datasets
    )


def choose_latest_runs_by_chunk(
    eval_results_root: Path,
    expected_datasets: list[str],
) -> list[RunInfo]:
    selected: dict[int, RunInfo] = {}
    expected_dataset_set = set(expected_datasets)

    for run_dir in sorted(eval_results_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info = parse_run_info(run_dir)
        if run_info is None:
            continue
        if set(run_info.datasets) != expected_dataset_set:
            continue
        if not _run_has_expected_outputs(run_dir, expected_datasets):
            continue

        current = selected.get(run_info.chunk_size)
        if current is None:
            selected[run_info.chunk_size] = run_info
            continue

        current_key = (int(current.has_dense_logging), current.run_name)
        candidate_key = (int(run_info.has_dense_logging), run_info.run_name)
        if candidate_key > current_key:
            selected[run_info.chunk_size] = run_info

    return [selected[chunk_size] for chunk_size in sorted(selected)]


def choose_latest_runs_by_dataset_and_chunk(
    eval_results_root: Path,
    expected_datasets: list[str],
) -> dict[str, list[RunInfo]]:
    selected: dict[tuple[str, int], RunInfo] = {}
    expected_dataset_set = set(expected_datasets)

    for run_dir in sorted(eval_results_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info = parse_run_info(run_dir)
        if run_info is None or len(run_info.datasets) != 1:
            continue

        dataset_name = run_info.datasets[0]
        if dataset_name not in expected_dataset_set:
            continue
        if not _run_has_expected_outputs(run_dir, [dataset_name]):
            continue

        key = (dataset_name, run_info.chunk_size)
        current = selected.get(key)
        if current is None:
            selected[key] = run_info
            continue

        current_key = (int(current.has_dense_logging), current.run_name)
        candidate_key = (int(run_info.has_dense_logging), run_info.run_name)
        if candidate_key > current_key:
            selected[key] = run_info

    grouped: dict[str, list[RunInfo]] = {
        dataset_name: [] for dataset_name in expected_datasets
    }
    for (dataset_name, _), run_info in selected.items():
        grouped[dataset_name].append(run_info)

    for dataset_name, run_infos in grouped.items():
        grouped[dataset_name] = sorted(
            run_infos, key=lambda run_info: run_info.chunk_size
        )

    return grouped


def choose_latest_runs_by_dataset_chunk_and_overlap(
    eval_results_root: Path,
    expected_datasets: list[str],
    ctx_chunk_overlap: int,
) -> dict[str, list[RunInfo]]:
    selected: dict[tuple[str, int], RunInfo] = {}
    expected_dataset_set = set(expected_datasets)

    for run_dir in sorted(eval_results_root.iterdir()):
        if not run_dir.is_dir():
            continue

        run_info = parse_run_info(run_dir)
        if run_info is None or len(run_info.datasets) != 1:
            continue
        if run_info.ctx_chunk_overlap != ctx_chunk_overlap:
            continue

        dataset_name = run_info.datasets[0]
        if dataset_name not in expected_dataset_set:
            continue
        if not _run_has_expected_outputs(run_dir, [dataset_name]):
            continue

        key = (dataset_name, run_info.chunk_size)
        current = selected.get(key)
        if current is None:
            selected[key] = run_info
            continue

        current_key = (int(current.has_dense_logging), current.run_name)
        candidate_key = (int(run_info.has_dense_logging), run_info.run_name)
        if candidate_key > current_key:
            selected[key] = run_info

    grouped: dict[str, list[RunInfo]] = {
        dataset_name: [] for dataset_name in expected_datasets
    }
    for (dataset_name, _), run_info in selected.items():
        grouped[dataset_name].append(run_info)

    for dataset_name, run_infos in grouped.items():
        grouped[dataset_name] = sorted(
            run_infos, key=lambda run_info: run_info.chunk_size
        )

    return grouped


def load_results_json(run_dir: Path, dataset_name: str) -> dict[str, object]:
    path = run_dir / f"test_{dataset_name}_results.json"
    return json.loads(path.read_text(encoding="utf-8"))


def load_dense_stats_json(run_dir: Path, dataset_name: str) -> dict[str, object] | None:
    path = run_dir / f"test_{dataset_name}_dense_lora_magnitude.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def _to_float(value: object) -> float | None:
    if value in (None, "None", "N/A"):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    return None


def get_performance_value(
    run_dir: Path,
    dataset_name: str,
    metric_name: str = "rougeL.f1",
) -> float | None:
    result_json = load_results_json(run_dir, dataset_name)
    return _to_float(result_json.get(f"test_{dataset_name}_{metric_name}"))


def get_dense_value(
    run_dir: Path,
    dataset_name: str,
    metric_name: str = "mean",
) -> float | None:
    result_json = load_results_json(run_dir, dataset_name)
    metric_key = f"test_{dataset_name}_dense_lora_magnitude_{metric_name}"
    value = _to_float(result_json.get(metric_key))
    if value is not None:
        return value

    dense_stats = load_dense_stats_json(run_dir, dataset_name)
    if dense_stats is None:
        return None

    overall = dense_stats.get("overall", {})
    if not isinstance(overall, dict):
        return None
    return _to_float(overall.get(metric_name))


def get_dense_values(
    run_dir: Path,
    dataset_name: str,
    metric_names: list[str] | tuple[str, ...],
) -> dict[str, float | None]:
    return {
        metric_name: get_dense_value(
            run_dir,
            dataset_name,
            metric_name=metric_name,
        )
        for metric_name in metric_names
    }


def ensure_output_path(output_path: Path) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    return output_path
