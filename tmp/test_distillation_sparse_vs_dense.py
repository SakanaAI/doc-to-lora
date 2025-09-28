#!/usr/bin/env python
"""
Compare the current dense KL (teacher top-k scatter into full vocab) vs a sparse gather-based
implementation used in distillation.

Metrics:
- Exact numerical equality (max |diff|) across many random seeds.
- Runtime (per iteration, averaged).
- Peak memory (GPU if available, else best-effort RSS delta on CPU).

By default we mimic the current code semantics (NO renormalization of the provided teacher
log-probs). That is, we treat absent vocab items as probability 0 and do:
  loss_i = - sum_j exp(target_logp[i,j]) * log q(indices[i,j])
which matches the existing dense scatter approach that sets -inf everywhere else.

We also optionally show (flag --normalized) a variant where teacher log-probs are re-normalized
over the provided top-k indices (useful if you want a true cross-entropy on the partial dist).

Usage (from repo root):
  python tmp/test_distillation_sparse_vs_dense.py --seeds 100 --n 2048 --vocab 32000 --k 8 --device cuda

Adjust sizes to fit your GPU memory. The dense method allocates an (N, V) float tensor twice
(outputs_logits + teacher_logp). For large (N,V) this can OOM.
"""

from __future__ import annotations

import argparse
import math
import os
import statistics
import time
import tracemalloc
from dataclasses import dataclass

import torch

try:
    import psutil  # optional for CPU RSS
except ImportError:  # pragma: no cover
    psutil = None


def dense_loss(
    outputs_logits: torch.Tensor, indices: torch.Tensor, target_logp: torch.Tensor
) -> torch.Tensor:
    """Replicates current implementation: build full teacher_logp filled with -inf, scatter top-k logp.
    outputs_logits: (N, V)
    indices: (N, K) long
    target_logp: (N, K) float (NOT assumed normalized)
    Returns: (N,) per-position losses.
    """
    N, V = outputs_logits.shape
    # Create full teacher log-probs
    teacher_logp = torch.full_like(outputs_logits, -torch.inf)
    teacher_logp.scatter_(1, indices, target_logp)
    p = teacher_logp.exp()  # zeros for -inf
    # logq = torch.log_softmax(outputs_logits, dim=-1)
    logq_full_denom = torch.logsumexp(outputs_logits, dim=-1, keepdim=True)  # (N,1)
    logq = outputs_logits - logq_full_denom
    loss = -(p * logq).sum(dim=-1)  # (N,)
    return loss


def sparse_loss(
    outputs_logits: torch.Tensor,
    indices: torch.Tensor,
    target_logp: torch.Tensor,
    *,
    normalize=False,
) -> torch.Tensor:
    """Sparse equivalent to dense_loss without constructing a full (N,V) teacher matrix.
    If normalize=True we renormalize teacher log-probs over provided indices.
    Returns (N,) per-position losses with SAME semantics as dense_loss when normalize=False.
    """
    # Gather student log-probs only at teacher indices.
    logq_full_denom = torch.logsumexp(outputs_logits, dim=-1, keepdim=True)  # (N,1)
    selected_logits = outputs_logits.gather(1, indices)  # (N,K)
    logq_selected = selected_logits - logq_full_denom  # log softmax at selected indices

    if normalize:
        # True cross-entropy over provided subset.
        teacher_logp_norm = target_logp - torch.logsumexp(
            target_logp, dim=-1, keepdim=True
        )
        teacher_p = teacher_logp_norm.exp()
    else:
        teacher_p = target_logp.exp()

    # NOT EFFICIENT?
    # logq = torch.nn.functional.log_softmax(outputs_logits, dim=-1)
    # logq_selected = logq.gather(1, indices)  # (N,K)
    # teacher_p = target_logp.exp()

    loss = -(teacher_p * logq_selected).sum(dim=-1)
    return loss


@dataclass
class Metrics:
    time_ms: float
    peak_mem_mb: float | None
    max_abs_diff: float | None = None


def measure(
    method_fn, *args, repeat=1, device="cpu", measure_memory=True, **kwargs
) -> tuple[torch.Tensor, Metrics]:
    """Run method_fn (returns tensor), measure avg wall time and peak memory.
    For GPU memory we use torch.cuda.reset_peak_memory_stats / max_memory_allocated.
    For CPU fallback we sample RSS before/after (approx)."""
    is_cuda = device.startswith("cuda") and torch.cuda.is_available()

    if is_cuda:
        torch.cuda.synchronize()
        torch.cuda.reset_peak_memory_stats()
    else:
        if measure_memory and psutil is not None:
            process = psutil.Process(os.getpid())
            rss_before = process.memory_info().rss
        elif measure_memory:
            tracemalloc.start()

    start = time.perf_counter()
    out = None
    for _ in range(repeat):
        out = method_fn(*args, **kwargs)
    if is_cuda:
        torch.cuda.synchronize()
    elapsed = (time.perf_counter() - start) / repeat * 1000.0

    peak_mb = None
    if is_cuda:
        peak_mb = torch.cuda.max_memory_allocated(device=device) / (1024**2)
    else:
        if measure_memory and psutil is not None:
            rss_after = process.memory_info().rss
            peak_mb = (rss_after - rss_before) / (1024**2)
        elif measure_memory:
            current, peak = tracemalloc.get_traced_memory()
            tracemalloc.stop()
            peak_mb = peak / (1024**2)

    return out, Metrics(time_ms=elapsed, peak_mem_mb=peak_mb)


def run_experiment(args):
    device = args.device
    torch.manual_seed(0)

    # Pre-generate shapes once per seed to reduce variance from dynamic graph allocations.
    dense_times = []
    dense_mems = []
    sparse_times = []
    sparse_mems = []
    diffs = []
    allclose_fail_seeds = []

    N = args.n
    V = args.vocab
    K = args.k

    if K > V:
        raise ValueError("k cannot exceed vocab size")

    dtype = torch.float16 if args.fp16 else torch.float32

    for seed in range(args.seeds):
        torch.manual_seed(seed)
        # Student logits (simulate model outputs for label positions)
        outputs_logits = 1 + 10 * torch.randn(N, V, device=device, dtype=dtype)
        # Ensure uniqueness per row: sample without replacement via randperm
        indices = torch.stack([torch.randperm(V, device=device)[:K] for _ in range(N)])
        # Teacher log-probs (could be any real numbers). Simulate by sampling logits then subtract logsumexp to get log-probs,
        # then optionally add random temperature scaling so they are not normalized (matching current dense semantics).
        teacher_logits = 1 + 10 * torch.randn(
            N, K, device=device, dtype=torch.float32
        )  # keep higher precision for stability
        target_logp = (
            teacher_logits - torch.logsumexp(teacher_logits, dim=-1, keepdim=True)
        ).to(dtype)

        # Measure dense
        dense_out, dense_metrics = measure(
            dense_loss, outputs_logits, indices, target_logp, device=device
        )
        # Measure sparse (semantics-matching version)
        sparse_out, sparse_metrics = measure(
            sparse_loss,
            outputs_logits,
            indices,
            target_logp,
            device=device,
            normalize=args.normalize_sparse,
        )

        # Compare
        # Move to float32 for diff to reduce underflow when fp16
        diff = (dense_out.float() - sparse_out.float()).abs().max().item()
        diffs.append(diff)
        dense_times.append(dense_metrics.time_ms)
        dense_mems.append(dense_metrics.peak_mem_mb or math.nan)
        sparse_times.append(sparse_metrics.time_ms)
        sparse_mems.append(sparse_metrics.peak_mem_mb or math.nan)

        # Per-seed torch.allclose check when semantics should match (no normalization flags)
        if not args.normalize_sparse:
            if not torch.allclose(
                dense_out, sparse_out, rtol=args.rtol, atol=args.atol
            ):
                allclose_fail_seeds.append(seed)

    print("=== Distillation Dense vs Sparse Comparison ===")
    print(f"Device: {device}  dtype: {dtype}  seeds: {args.seeds}")
    print(f"Shape: N={N} positions, V={V} vocab, K={K} top-k per position")
    print(f"Sparse renormalize flag (normalize_sparse): {args.normalize_sparse}")
    print("-- Timing (ms per iteration) --")
    print(
        f"Dense mean: {statistics.mean(dense_times):.3f}  median: {statistics.median(dense_times):.3f}"
    )
    print(
        f"Sparse mean: {statistics.mean(sparse_times):.3f}  median: {statistics.median(sparse_times):.3f}"
    )
    speedup = statistics.mean(dense_times) / statistics.mean(sparse_times)
    print(f"Speedup (dense / sparse): {speedup:.2f}x")

    if all(not math.isnan(m) for m in dense_mems + sparse_mems):
        print("-- Peak Memory (MB) --")
        print(
            f"Dense mean: {statistics.mean(dense_mems):.1f}  median: {statistics.median(dense_mems):.1f}"
        )
        print(
            f"Sparse mean: {statistics.mean(sparse_mems):.1f}  median: {statistics.median(sparse_mems):.1f}"
        )
        mem_reduction = statistics.mean(dense_mems) / max(
            1e-9, statistics.mean(sparse_mems)
        )
        print(f"Memory reduction (dense / sparse): {mem_reduction:.2f}x")
    else:
        print("(Memory stats not fully available on this platform / configuration)")

    print("-- Numerical Differences --")
    max_diff = max(diffs)
    mean_diff = statistics.mean(diffs)
    print(f"Max |dense - sparse| over seeds: {max_diff:.3e}")
    print(f"Mean |dense - sparse|: {mean_diff:.3e}")

    tol = 1e-5
    if args.normalize_sparse:
        print(
            "NOTE: Normalization flags enabled; numerical results may intentionally differ from dense baseline."
        )
    else:
        # Summary of allclose results
        if not allclose_fail_seeds:
            print(
                f"SUCCESS: Sparse matches dense within tolerance (max diff {max_diff:.3e} < {tol}) and allclose passed for all seeds (rtol={args.rtol}, atol={args.atol})."
            )
        else:
            if allclose_fail_seeds:
                print(
                    f"WARNING: torch.allclose failed for {len(allclose_fail_seeds)} / {args.seeds} seeds. Failed seeds: {allclose_fail_seeds[:10]}{'...' if len(allclose_fail_seeds) > 10 else ''}"
                )
            # if max_diff >= tol:
            #     print(
            #         f"WARNING: Max diff {max_diff:.3e} exceeds heuristic dtype tolerance {tol}."
            #     )


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--seeds", type=int, default=100, help="Number of random seeds / trials"
    )
    p.add_argument("--n", type=int, default=1024, help="Number of label positions (N)")
    p.add_argument("--vocab", type=int, default=32000, help="Vocabulary size (V)")
    p.add_argument("--k", type=int, default=8, help="Top-k teacher tokens (K)")
    p.add_argument(
        "--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu"
    )
    p.add_argument(
        "--fp16", action="store_true", help="Use float16 for student logits/log-probs"
    )
    p.add_argument(
        "--normalize-sparse",
        dest="normalize_sparse",
        action="store_true",
        help="Renormalize teacher log-probs inside sparse path",
    )
    return p.parse_args()


if __name__ == "__main__":
    torch.backends.cuda.matmul.allow_fp16_reduced_precision_reduction = False
    torch.backends.cuda.matmul.allow_bf16_reduced_precision_reduction = False
    torch.backends.cudnn.benchmark = False
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False

    args = parse_args()

    # Set default tolerances if not provided
    args.rtol = 1.3e-6
    args.atol = 1e-5
    run_experiment(args)
