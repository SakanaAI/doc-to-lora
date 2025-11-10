"""Unified tracking interface combining timing and CUDA memory usage.

Primary API
-----------
add_tracker(bound_method, name)
    Wraps a bound instance method so that each invocation records:
      - wall-clock duration (seconds) in timer.TIMER_REGISTRY[name]
      - CUDA peak memory increase (bytes) in cuda_memory_tracker.MEMORY_REGISTRY[name]
        (only if CUDA + torch available; otherwise memory list may stay empty / absent)

print_tracker_stats(name=None)
    Convenience printer that delegates to time + memory aggregate printers.

Design
------
We implement a single wrapper (instead of nesting the individual timer & memory
wrappers) to avoid multiple layers of indirection and to ensure the measured
CUDA memory footprint reflects only the original method's body (excluding the
separate timing wrapper's slight overhead). The wrapper is idempotent: repeated
calls to add_tracker on the same method are ignored.

This file depends on sibling modules:
- tracker.timer
- tracker.cuda_memory_tracker

Both registries remain the single source of truth; no additional registry is introduced.
"""

from __future__ import annotations

from collections.abc import Callable
from time import perf_counter
from typing import Any

# Support both package (relative) and direct script execution.
try:  # Package / normal import path
    from .cuda_memory_tracker import (  # type: ignore
        MEMORY_REGISTRY,
        compute_aggregate_memory_stats,
        print_aggregate_memory_stats,
        print_global_memory_stats,
        reset_memory_trackers,
        save_memory_stats_csv,
    )
    from .timer import (  # type: ignore
        TIMER_REGISTRY,
        compute_aggregate_timer_stats,
        print_aggregate_timer_stats,
        print_global_timer_stats,
        reset_timers,
        save_timer_stats_csv,
    )
except Exception:  # pragma: no cover - fallback when executed directly
    import pathlib
    import sys

    _this_file = pathlib.Path(__file__).resolve()
    # project root is two levels up from tracker/ (i.e., .../src)
    _src_root = _this_file.parents[2]
    if str(_src_root) not in sys.path:
        sys.path.insert(0, str(_src_root))
    try:
        from ctx_to_lora.tracker.cuda_memory_tracker import (  # type: ignore
            MEMORY_REGISTRY,
            compute_aggregate_memory_stats,
            print_aggregate_memory_stats,
            print_global_memory_stats,
            reset_memory_trackers,
            save_memory_stats_csv,
        )
        from ctx_to_lora.tracker.timer import (  # type: ignore
            TIMER_REGISTRY,
            compute_aggregate_timer_stats,
            print_aggregate_timer_stats,
            print_global_timer_stats,
            reset_timers,
            save_timer_stats_csv,
        )
    except Exception as e:  # If still failing, raise a clearer error.
        raise ImportError(
            f"Failed to import tracking dependencies; ensure project root on PYTHONPATH. Original: {e}"
        )

try:  # Optional torch import (lazy fallback if unavailable)
    import torch  # type: ignore
except Exception:  # pragma: no cover - torch absence path
    torch = None  # type: ignore

__all__ = [
    "add_tracker",
    "compute_tracker_stats",
    "save_tracker_stats_csv",
    "print_tracker_stats",
    "print_global_tracker_stats",
    "reset_trackers",
]


def _cuda_available() -> bool:
    return bool(torch is not None and torch.cuda.is_available())


def add_tracker(func: Callable, name: str) -> None:
    """Attach a combined time + CUDA memory tracking wrapper to a bound method.

    Parameters
    ----------
    func : Callable
        A bound instance method (instance.method). Raises ValueError if unbound.
    name : str
        Registry key under which metrics are stored.
    """
    if not hasattr(func, "__self__") or getattr(func, "__self__") is None:
        # Permit idempotent re-calls if already wrapped.
        if getattr(func, "__is_tracker_wrapper__", False):
            return
        raise ValueError(
            "add_tracker expects a bound method: call with instance.method"
        )

    instance = func.__self__  # underlying object
    method_name = getattr(func, "__name__", None)
    if method_name is None:
        raise ValueError("Cannot determine method name for provided callable")

    existing = getattr(instance, method_name, None)
    if getattr(
        existing, "__is_tracker_wrapper__", False
    ):  # Already wrapped via unified tracker
        return

    # If already individually wrapped by timer or memory tracker, we still wrap only once more;
    # future calls to add_tracker will become no-ops.
    orig_bound = existing if existing is not None else func

    def tracked(*args: Any, **kwargs: Any):  # noqa: D401 - combined wrapper
        use_cuda = _cuda_available()
        if use_cuda:
            torch.cuda.synchronize()
            torch.cuda.reset_peak_memory_stats()
            start_alloc = torch.cuda.memory_allocated()
        start_time = perf_counter()
        try:
            return orig_bound(*args, **kwargs)
        finally:
            elapsed = perf_counter() - start_time
            TIMER_REGISTRY.setdefault(name, []).append(elapsed)
            if use_cuda:
                torch.cuda.synchronize()
                peak_alloc = torch.cuda.max_memory_allocated()
                peak_increase = peak_alloc - start_alloc
                if peak_increase < 0:  # Safety guard (should not happen)
                    peak_increase = 0
                MEMORY_REGISTRY.setdefault(name, []).append(int(peak_increase))

    # Introspection / idempotency markers
    tracked.__name__ = method_name
    tracked.__qualname__ = getattr(orig_bound, "__qualname__", method_name)
    tracked.__doc__ = getattr(orig_bound, "__doc__")
    tracked.__wrapped__ = orig_bound  # type: ignore[attr-defined]
    tracked.__is_tracker_wrapper__ = True  # type: ignore[attr-defined]
    tracked.__is_timer_wrapper__ = True  # type: ignore[attr-defined]
    tracked.__is_memory_wrapper__ = True  # type: ignore[attr-defined]
    tracked.__tracker_name__ = name  # type: ignore[attr-defined]

    setattr(instance, method_name, tracked)


def compute_tracker_stats(
    name: str | None = None,
) -> dict[str, dict[str, Any]] | None:
    """Compute both timing and memory stats for a given name (or all if None).

    Parameters
    ----------
    name : Optional[str]
        Specific tracker name; if None, computes all.

    Returns
    -------
    Optional[Dict[str, Dict[str, Any]]]
        None if no data, else dict with 'timing' and 'memory' keys containing
        their respective aggregate statistics.
    """
    timer_stats = compute_aggregate_timer_stats(name)
    memory_stats = compute_aggregate_memory_stats(name)

    if timer_stats is None and memory_stats is None:
        return None

    return {
        "timing": timer_stats or {},
        "memory": memory_stats or {},
    }


def save_tracker_stats_csv(file_path: str, name: str | None = None) -> None:
    """Save both timing and memory stats to separate CSV files.

    Parameters
    ----------
    file_path : str
        Base path for CSV files. Will create file_path_timing.csv and file_path_memory.csv
    name : Optional[str]
        Specific tracker name to export. If None, all trackers are exported.
    """
    import os

    os.makedirs(os.path.dirname(file_path), exist_ok=True)

    base_path = os.path.splitext(file_path)[0]
    timer_path = f"{base_path}_timing.csv"
    memory_path = f"{base_path}_memory.csv"

    # Save timing stats if available
    timer_stats = compute_aggregate_timer_stats(name)
    if timer_stats is not None:
        save_timer_stats_csv(timer_path, name)

    # Save memory stats if available
    memory_stats = compute_aggregate_memory_stats(name)
    if memory_stats is not None:
        save_memory_stats_csv(memory_path, name)

    # If no data at all, raise an error
    if timer_stats is None and memory_stats is None:
        print("No tracking data available to export")


def print_tracker_stats(name: str | None = None) -> None:
    """Print both timing and memory stats for a given name (or all if None).

    Parameters
    ----------
    name : Optional[str]
        Specific tracker name; if None, prints all.
    """
    print("[tracker] Timing stats:")
    print_aggregate_timer_stats(name)
    print("\n[tracker] CUDA memory stats:")
    print_aggregate_memory_stats(name)


def print_global_tracker_stats() -> None:
    """Print global aggregate timing and memory stats."""
    print("[tracker] Global timing stats:")
    print_global_timer_stats()
    print("\n[tracker] Global CUDA memory stats:")
    print_global_memory_stats()


def reset_trackers() -> None:
    """Reset all timer and memory tracking data."""
    reset_timers()
    reset_memory_trackers()


if __name__ == "__main__":  # Demonstration
    import random
    import time

    class Demo:
        def compute(self, n: int = 25_000) -> int:
            # CPU-bound work
            s = 0
            for i in range(n):
                s += i * i
            return s

        def gpu_alloc(self, n: int = 500_000):
            if not _cuda_available():
                # Simulate light wait to differentiate timing
                time.sleep(random.uniform(0.01, 0.05))
                return None
            t = torch.empty(n, dtype=torch.float32, device="cuda")
            t.uniform_()  # ensure usage
            return t.sum().item()

    demo = Demo()

    add_tracker(demo.compute, "compute")
    add_tracker(demo.gpu_alloc, "gpu_alloc")
    # Idempotent re-call
    add_tracker(demo.compute, "compute")

    for _ in range(5):
        demo.compute(15_000)
        demo.gpu_alloc(300_000)

    print_tracker_stats()
    print("\n--- Global Combined Stats ---\n")
    print_global_tracker_stats()

    # Demonstrate CSV export
    print("\n[tracker] Saving stats to CSV files...\n")
    csv_path = "/tmp/tracker_demo_stats.csv"
    save_tracker_stats_csv(csv_path)
    print(f"Exported timing stats to: {csv_path.replace('.csv', '_timing.csv')}")
    print(f"Exported memory stats to: {csv_path.replace('.csv', '_memory.csv')}")

    print("\n[tracker] Resetting registries...\n")
    reset_trackers()
    print_tracker_stats()
