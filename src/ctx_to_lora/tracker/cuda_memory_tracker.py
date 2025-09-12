"""Lightweight CUDA memory tracking utilities for measuring per-method peak memory usage.

Usage:
    x = SomeClass()
    add_memory_tracker(x.some_method, "some_method")  # wraps the bound method in-place
    x.some_method(...)
    print_aggregate_memory_stats("some_method")

Design notes:
- Mirrors the API of timer.py but records CUDA memory (peak increase in bytes) per call.
- add_memory_tracker mutates the instance method with a wrapper (idempotent: double wrap avoided).
- Global registry: { name: [int, ...] } storing per-call peak memory increase (bytes).
- print_aggregate_memory_stats prints summary stats (count, total, mean, median, min, max, p95, last).
- If CUDA or torch is unavailable, wrappers degrade gracefully (no measurements recorded).

Metrics collected per call (if CUDA available):
- peak_increase_bytes: (torch.cuda.max_memory_allocated() - start_allocated)
  This captures the maximum additional memory pressure during the call.

Caveats:
- Rapid allocate/free patterns entirely inside the call still reflect peak transient usage.
- Asynchronous CUDA ops: we synchronize before and after to improve accuracy. This may
  slightly affect performance timings but is necessary for memory correctness.
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, median, stdev
from typing import Any

try:  # Optional dependency handling
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch absence path
    torch = None  # type: ignore

# Global memory registry: name -> list of peak memory increases (bytes)
MEMORY_REGISTRY: dict[str, list[int]] = {}


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def add_memory_tracker(func: Callable, name: str) -> None:
    """Attach a CUDA memory tracking wrapper to a bound method.

    Parameters
    ----------
    func : Callable
        A *bound* instance method (instance.method). Raises ValueError if unbound.
    name : str
        Key under which memory stats are recorded in MEMORY_REGISTRY.
    """
    if not hasattr(func, "__self__") or getattr(func, "__self__") is None:
        if getattr(func, "__is_memory_wrapper__", False):  # already wrapped
            return
        raise ValueError(
            "add_memory_tracker expects a bound method: call with instance.method"
        )

    instance = func.__self__
    method_name = getattr(func, "__name__", None)
    if method_name is None:
        raise ValueError("Cannot determine method name for provided callable")

    existing = getattr(instance, method_name, None)
    if getattr(existing, "__is_memory_wrapper__", False):  # idempotent
        return

    orig_bound = func

    def tracked(*args: Any, **kwargs: Any):  # noqa: D401 - simple wrapper
        if not _cuda_available():
            return orig_bound(*args, **kwargs)
        # Synchronize to get a clean baseline
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
        start_alloc = torch.cuda.memory_allocated()
        try:
            return orig_bound(*args, **kwargs)
        finally:
            torch.cuda.synchronize()
            peak_alloc = torch.cuda.max_memory_allocated()
            peak_increase = peak_alloc - start_alloc
            # Record only if positive (avoid negative due to potential race, though improbable)
            if peak_increase < 0:
                peak_increase = 0
            MEMORY_REGISTRY.setdefault(name, []).append(int(peak_increase))

    tracked.__name__ = method_name
    tracked.__doc__ = getattr(orig_bound, "__doc__")
    tracked.__qualname__ = getattr(orig_bound, "__qualname__", method_name)
    tracked.__is_memory_wrapper__ = True  # type: ignore[attr-defined]
    tracked.__wrapped__ = orig_bound  # type: ignore[attr-defined]
    tracked.__memory_name__ = name  # type: ignore[attr-defined]

    setattr(instance, method_name, tracked)


def _format_bytes(num_bytes: float) -> str:
    """Human-readable byte formatting (base-2)."""
    if num_bytes < 1024:
        return f"{int(num_bytes):5d}B"
    units = ["KiB", "MiB", "GiB", "TiB"]
    value = float(num_bytes)
    for u in units:
        value /= 1024.0
        if value < 1024.0:
            return f"{value:8.3f}{u}"
    return f"{value:8.3f}PiB"  # Extremely unlikely


def compute_aggregate_memory_stats(
    name: str | None = None,
) -> dict[str, dict[str, float]] | None:
    """Compute aggregate CUDA memory statistics for specific trackers.

    Parameters
    ----------
    name : Optional[str]
        Specific tracker name to compute stats for. If None, all trackers are computed.

    Returns
    -------
    Optional[Dict[str, Dict[str, float]]]
        None if no data, else dict mapping tracker names to their stats.
        Each stats dict contains: count, total, mean, median, min, max, p95, last, std.
    """
    if not MEMORY_REGISTRY:
        return None

    keys = [name] if name else sorted(MEMORY_REGISTRY.keys())
    valid_keys = [k for k in keys if k in MEMORY_REGISTRY and MEMORY_REGISTRY[k]]
    if not valid_keys:
        return None

    result = {}
    for k in valid_keys:
        data = MEMORY_REGISTRY[k]
        data_sorted = sorted(data)
        cnt = len(data)
        total = sum(data)
        avg = mean(data)
        med = median(data)
        std = stdev(data) if cnt > 1 else 0.0
        mn = data_sorted[0]
        mx = data_sorted[-1]
        p95_index = int(0.95 * (cnt - 1))
        p95 = data_sorted[p95_index]
        last = data[-1]
        result[k] = {
            "count": float(cnt),
            "total": float(total),
            "mean": float(avg),
            "median": float(med),
            "min": float(mn),
            "max": float(mx),
            "p95": float(p95),
            "last": float(last),
            "std": float(std),
        }
    return result


def save_memory_stats_csv(file_path: str, name: str | None = None) -> None:
    """Save aggregate CUDA memory statistics to a CSV file.

    Parameters
    ----------
    file_path : str
        Path where the CSV file will be saved.
    name : Optional[str]
        Specific tracker name to export. If None, all trackers are exported.
    """
    import csv

    stats = compute_aggregate_memory_stats(name)
    if stats is None:
        raise ValueError("No memory data available to export")

    with open(file_path, "w", newline="") as csvfile:
        fieldnames = [
            "name",
            "count",
            "total",
            "mean",
            "median",
            "min",
            "max",
            "p95",
            "last",
            "std",
        ]
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()

        for tracker_name, data in stats.items():
            row = {"name": tracker_name}
            row.update(data)
            writer.writerow(row)


def print_aggregate_memory_stats(name: str | None = None) -> None:
    """Print aggregate CUDA memory stats for one or all tracked names.

    Parameters
    ----------
    name : Optional[str]
        Specific name to report; if None, report all.
    """
    stats = compute_aggregate_memory_stats(name)
    if stats is None:
        print("[mem] No memory data collected.")
        return

    keys = [name] if name else sorted(MEMORY_REGISTRY.keys())
    missing = [k for k in keys if k not in MEMORY_REGISTRY or not MEMORY_REGISTRY[k]]
    if missing:
        print(f"[mem] No data for: {', '.join(missing)}")

    if not stats:
        return

    header = (
        f"{'name':20} {'count':>6} {'total':>12} {'mean':>12} {'median':>12} "
        f"{'min':>12} {'max':>12} {'p95':>12} {'std':>12} {'last':>12}"
    )
    print(header)
    print("-" * len(header))

    for k, data in stats.items():
        print(
            f"{k:20} {int(data['count']):6d} {_format_bytes(data['total']):>12} "
            f"{_format_bytes(data['mean']):>12} {_format_bytes(data['median']):>12} "
            f"{_format_bytes(data['min']):>12} {_format_bytes(data['max']):>12} "
            f"{_format_bytes(data['p95']):>12} {_format_bytes(data['std']):>12} "
            f"{_format_bytes(data['last']):>12}"
        )


def compute_global_memory_stats() -> dict[str, float] | None:
    """Compute aggregate stats across all recorded memory entries.

    Returns
    -------
    Optional[Dict[str, float]]
        None if no data; else dict with count, total, mean, median, min, max, p95, std.
    """
    if not MEMORY_REGISTRY:
        return None
    all_values: list[int] = []
    for lst in MEMORY_REGISTRY.values():
        all_values.extend(lst)
    if not all_values:
        return None
    data_sorted = sorted(all_values)
    cnt = len(all_values)
    total = float(sum(all_values))
    avg = mean(all_values)
    med = median(all_values)
    std = stdev(all_values) if cnt > 1 else 0.0
    mn = float(data_sorted[0])
    mx = float(data_sorted[-1])
    p95_index = int(0.95 * (cnt - 1))
    p95 = float(data_sorted[p95_index])
    return {
        "count": float(cnt),
        "total": total,
        "mean": float(avg),
        "median": float(med),
        "min": mn,
        "max": mx,
        "p95": p95,
        "std": float(std),
    }


def print_global_memory_stats() -> None:
    """Pretty-print global stats across all memory registry entries."""
    stats = compute_global_memory_stats()
    if stats is None:
        print("[mem] No memory data collected.")
        return
    header = (
        f"{'scope':20} {'count':>6} {'total':>12} {'mean':>12} {'median':>12} "
        f"{'min':>12} {'max':>12} {'p95':>12} {'std':>12}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'<ALL>':20} {int(stats['count']):6d} {_format_bytes(stats['total']):>12} "
        f"{_format_bytes(stats['mean']):>12} {_format_bytes(stats['median']):>12} "
        f"{_format_bytes(stats['min']):>12} {_format_bytes(stats['max']):>12} "
        f"{_format_bytes(stats['p95']):>12} {_format_bytes(stats['std']):>12}"
    )


def reset_memory_trackers() -> None:
    """Clear all recorded memory tracking data."""
    MEMORY_REGISTRY.clear()


__all__ = [
    "MEMORY_REGISTRY",
    "add_memory_tracker",
    "compute_aggregate_memory_stats",
    "save_memory_stats_csv",
    "print_aggregate_memory_stats",
    "compute_global_memory_stats",
    "print_global_memory_stats",
    "reset_memory_trackers",
]


if __name__ == "__main__":  # Simple demonstration

    class Demo:
        def __init__(self, device: str | None = None):
            self.device = device or ("cuda" if _cuda_available() else "cpu")

        def allocate(self, n: int = 1_000_000) -> int:
            if not _cuda_available():
                # Fallback: just create a CPU tensor
                _ = [0] * n  # noqa: F841
                return n
            import torch  # local import to avoid mypy confusion

            t = torch.empty(n, dtype=torch.float32, device=self.device)
            # Perform an op to ensure allocation
            t.uniform_()  # noqa: F841
            return t.numel()

        def noalloc(self):  # method with negligible allocation
            return 42

    demo = Demo()
    add_memory_tracker(demo.allocate, "alloc")
    add_memory_tracker(demo.noalloc, "noalloc")
    add_memory_tracker(demo.allocate, "alloc")  # idempotent

    for _ in range(5):
        demo.allocate(200_000)
        demo.noalloc()

    print("\nAll memory stats:\n")
    print_aggregate_memory_stats()

    print("\nSingle (alloc):\n")
    print_aggregate_memory_stats("alloc")

    print("\nGlobal memory stats:\n")
    print_global_memory_stats()
