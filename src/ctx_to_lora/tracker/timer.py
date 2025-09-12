"""Lightweight timing utilities for attaching runtime measurement to object methods.

Usage:
    x = SomeClass()
    add_timer(x.some_method, "some_method")  # wraps the bound method in-place
    x.some_method(...)
    print_aggregate_timer_stats("some_method")

Design notes:
- add_timer mutates the instance by replacing the bound method with a timing wrapper.
- Multiple calls to add_timer on the same (already wrapped) method are ignored to avoid double timing.
- Global registry: { name: [float, ...] } storing individual call durations.
- print_aggregate_timer_stats prints summary stats (count, total, mean, median, min, max, p95, last).
"""

from __future__ import annotations

from collections.abc import Callable
from statistics import mean, median, stdev
from time import perf_counter
from typing import Any

# Global timer registry: name -> list of durations (seconds)
TIMER_REGISTRY: dict[str, list[float]] = {}


class _TimerWrapperMarker:
    """Mixin marker to identify already wrapped callables."""

    __slots__ = ("__wrapped_name__",)

    def __init__(self, wrapped_name: str):
        self.__wrapped_name__ = wrapped_name


def add_timer(func: Callable, name: str) -> None:
    """Attach a timing wrapper to a bound method.

    Parameters
    ----------
    func : Callable
        A *bound* method (e.g., instance.method). If an unbound function is provided
        it will raise a ValueError (explicitness keeps behavior predictable).
    name : str
        Key under which durations are recorded in TIMER_REGISTRY.
    """
    # Basic validation (permit already wrapped functions for idempotency even though
    # they sit as plain functions on the instance dict and thus lack __self__).
    if not hasattr(func, "__self__") or getattr(func, "__self__") is None:
        if getattr(func, "__is_timer_wrapper__", False):  # already wrapped, no-op
            return
        raise ValueError("add_timer expects a bound method: call with instance.method")

    instance = func.__self__  # The object instance
    method_name = getattr(func, "__name__", None)
    if method_name is None:
        raise ValueError("Cannot determine method name for provided callable")

    # Prevent double-wrapping (idempotent behavior)
    existing = getattr(instance, method_name, None)
    if getattr(existing, "__is_timer_wrapper__", False):  # Already wrapped
        return

    orig_bound = func  # capture original bound method

    def timed(*args: Any, **kwargs: Any):  # noqa: D401 - simple wrapper
        start = perf_counter()
        try:
            return orig_bound(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start
            TIMER_REGISTRY.setdefault(name, []).append(elapsed)

    # Mark wrapper to avoid double wrapping; preserve introspection hints.
    timed.__name__ = method_name
    timed.__doc__ = getattr(orig_bound, "__doc__")
    timed.__qualname__ = getattr(orig_bound, "__qualname__", method_name)
    timed.__is_timer_wrapper__ = True  # type: ignore[attr-defined]
    timed.__wrapped__ = orig_bound  # type: ignore[attr-defined]
    timed.__timer_name__ = name  # type: ignore[attr-defined]

    setattr(instance, method_name, timed)


def _format_seconds(sec: float) -> str:
    if sec >= 1:
        return f"{sec:8.3f}s"
    if sec >= 1e-3:
        return f"{sec * 1e3:8.3f}ms"
    if sec >= 1e-6:
        return f"{sec * 1e6:8.3f}µs"
    return f"{sec * 1e9:8.3f}ns"


def compute_aggregate_timer_stats(
    name: str | None = None,
) -> dict[str, dict[str, float]] | None:
    """Compute aggregate timing statistics for specific timers.

    Parameters
    ----------
    name : Optional[str]
        Specific timer name to compute stats for. If None, all timers are computed.

    Returns
    -------
    Optional[Dict[str, Dict[str, float]]]
        None if no data, else dict mapping timer names to their stats.
        Each stats dict contains: count, total, mean, median, min, max, p95, last, std.
    """
    if not TIMER_REGISTRY:
        return None

    keys = [name] if name else sorted(TIMER_REGISTRY.keys())
    valid_keys = [k for k in keys if k in TIMER_REGISTRY and TIMER_REGISTRY[k]]
    if not valid_keys:
        return None

    result = {}
    for k in valid_keys:
        data = TIMER_REGISTRY[k]
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
            "total": total,
            "mean": avg,
            "median": med,
            "min": mn,
            "max": mx,
            "p95": p95,
            "last": last,
            "std": std,
        }
    return result


def save_timer_stats_csv(file_path: str, name: str | None = None) -> None:
    """Save aggregate timing statistics to a CSV file.

    Parameters
    ----------
    file_path : str
        Path where the CSV file will be saved.
    name : Optional[str]
        Specific timer name to export. If None, all timers are exported.
    """
    import csv

    stats = compute_aggregate_timer_stats(name)
    if stats is None:
        raise ValueError("No timing data available to export")

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

        for timer_name, data in stats.items():
            row = {"name": timer_name}
            row.update(data)
            writer.writerow(row)


def print_aggregate_timer_stats(name: str | None = None) -> None:
    """Print aggregate timing statistics.

    Parameters
    ----------
    name : Optional[str]
        Specific timer name to report. If None, all timers are reported.
    """
    stats = compute_aggregate_timer_stats(name)
    if stats is None:
        print("[timer] No timing data collected.")
        return

    keys = [name] if name else sorted(TIMER_REGISTRY.keys())
    missing = [k for k in keys if k not in TIMER_REGISTRY or not TIMER_REGISTRY[k]]
    if missing:
        print(f"[timer] No data for: {', '.join(missing)}")

    if not stats:
        return

    header = (
        f"{'name':20} {'count':>6} {'total':>10} {'mean':>10} {'median':>10} "
        f"{'min':>10} {'max':>10} {'p95':>10} {'std':>10} {'last':>10}"
    )
    print(header)
    print("-" * len(header))

    for k, data in stats.items():
        print(
            f"{k:20} {int(data['count']):6d} {_format_seconds(data['total']):>10} "
            f"{_format_seconds(data['mean']):>10} {_format_seconds(data['median']):>10} "
            f"{_format_seconds(data['min']):>10} {_format_seconds(data['max']):>10} "
            f"{_format_seconds(data['p95']):>10} {_format_seconds(data['std']):>10} "
            f"{_format_seconds(data['last']):>10}"
        )


def compute_global_timer_stats() -> dict[str, float] | None:
    """Compute aggregate statistics across all recorded timer values.

    Returns
    -------
    Optional[Dict[str, float]]
        None if no data recorded, else a dict with keys:
        count, total, mean, median, min, max, p95, std.
    """
    if not TIMER_REGISTRY:
        return None
    all_values: list[float] = []
    for lst in TIMER_REGISTRY.values():
        all_values.extend(lst)
    if not all_values:
        return None
    data_sorted = sorted(all_values)
    cnt = len(all_values)
    total = sum(all_values)
    avg = mean(all_values)
    med = median(all_values)
    std = stdev(all_values) if cnt > 1 else 0.0
    mn = data_sorted[0]
    mx = data_sorted[-1]
    p95_index = int(0.95 * (cnt - 1))
    p95 = data_sorted[p95_index]
    return {
        "count": float(cnt),  # keep uniform numeric type
        "total": total,
        "mean": avg,
        "median": med,
        "min": mn,
        "max": mx,
        "p95": p95,
        "std": std,
    }


def print_global_timer_stats() -> None:
    """Pretty-print global aggregate stats across all timer entries."""
    stats = compute_global_timer_stats()
    if stats is None:
        print("[timer] No timing data collected.")
        return
    header = (
        f"{'scope':20} {'count':>6} {'total':>10} {'mean':>10} {'median':>10} "
        f"{'min':>10} {'max':>10} {'p95':>10} {'std':>10}"
    )
    print(header)
    print("-" * len(header))
    print(
        f"{'<ALL>':20} {int(stats['count']):6d} {_format_seconds(stats['total']):>10} "
        f"{_format_seconds(stats['mean']):>10} {_format_seconds(stats['median']):>10} "
        f"{_format_seconds(stats['min']):>10} {_format_seconds(stats['max']):>10} "
        f"{_format_seconds(stats['p95']):>10} {_format_seconds(stats['std']):>10}"
    )


def reset_timers() -> None:
    """Reset (clear) all recorded timing data."""
    TIMER_REGISTRY.clear()


__all__ = [
    "TIMER_REGISTRY",
    "add_timer",
    "compute_aggregate_timer_stats",
    "save_timer_stats_csv",
    "print_aggregate_timer_stats",
    "compute_global_timer_stats",
    "print_global_timer_stats",
    "reset_timers",
]

if __name__ == "__main__":
    # Example usage / simple self-test.
    import random
    import time

    class Demo:
        def f1(self, n: int = 20_000) -> int:
            # CPU-bound work
            s = 0
            for i in range(n):
                s += i * i
            return s

        def f2(self) -> None:
            # Simulate I/O or waiting
            time.sleep(random.uniform(0.001, 0.003))

    demo = Demo()

    # Attach timers (idempotent: calling again does nothing harmful)
    add_timer(demo.f1, "f1")
    add_timer(demo.f2, "f2")
    add_timer(demo.f1, "f1")  # demonstrate double-wrap prevention

    for _ in range(5):
        demo.f1(10_000)
        demo.f2()

    print("\nAll timers:\n")
    print_aggregate_timer_stats()

    print("\nSingle timer (f1):\n")
    print_aggregate_timer_stats("f1")

    print("\nGlobal aggregated stats:\n")
    print_global_timer_stats()
