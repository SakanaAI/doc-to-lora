from __future__ import annotations

import math
from dataclasses import dataclass, field

import torch
from torch import Tensor


@dataclass
class RunningMagnitudeStats:
    count: int = 0
    total: float = 0.0
    total_sq: float = 0.0
    min_val: float | None = None
    max_val: float | None = None

    def update(self, values: Tensor) -> None:
        if values.numel() == 0:
            return

        values = values.to(dtype=torch.float64)
        batch_count = int(values.numel())
        batch_sum = float(values.sum().item())
        batch_sum_sq = float(values.square().sum().item())
        batch_min = float(values.min().item())
        batch_max = float(values.max().item())

        self.count += batch_count
        self.total += batch_sum
        self.total_sq += batch_sum_sq
        self.min_val = (
            batch_min if self.min_val is None else min(self.min_val, batch_min)
        )
        self.max_val = (
            batch_max if self.max_val is None else max(self.max_val, batch_max)
        )

    def to_dict(self) -> dict[str, float | int | None]:
        if self.count == 0:
            return {
                "count": 0,
                "mean": None,
                "min": None,
                "max": None,
                "std": None,
            }

        mean = self.total / self.count
        variance = max(self.total_sq / self.count - mean**2, 0.0)
        return {
            "count": self.count,
            "mean": mean,
            "min": self.min_val,
            "max": self.max_val,
            "std": math.sqrt(variance),
        }


def _update_dense_magnitude_stats_for_module(
    stats: RunningMagnitudeStats | list[RunningMagnitudeStats],
    A: Tensor,
    B: Tensor,
    scaling: float,
    out_chunk_size: int,
) -> None:
    if A.ndim != 4 or B.ndim != 4:
        raise ValueError(
            f"Expected combined LoRA tensors with 4 dims, got {A.shape=} and {B.shape=}"
        )
    if A.shape[:3] != B.shape[:3]:
        raise ValueError(f"Mismatched LoRA shapes: {A.shape=} and {B.shape=}")
    if out_chunk_size <= 0:
        raise ValueError("`out_chunk_size` must be > 0.")
    if isinstance(stats, RunningMagnitudeStats):
        stats = [stats]
    if not torch.cuda.is_available():
        raise RuntimeError("Dense LoRA magnitude logging requires CUDA.")

    # Keep the entire computation on GPU to avoid CPU offloading.
    A_gpu = A.detach().to(device="cuda", dtype=torch.float32)
    B_gpu = B.detach().to(device="cuda", dtype=torch.float32)

    batch_size, num_layers, _, d_in = A_gpu.shape
    d_out = B_gpu.shape[-1]

    A_flat = A_gpu.reshape(batch_size * num_layers, A_gpu.shape[-2], d_in)
    B_flat = B_gpu.reshape(batch_size * num_layers, B_gpu.shape[-2], d_out)

    for start in range(0, d_out, out_chunk_size):
        stop = min(start + out_chunk_size, d_out)
        B_block = B_flat[:, :, start:stop].transpose(1, 2)
        dense_block = torch.bmm(B_block, A_flat)
        dense_block.mul_(float(scaling))
        abs_block = dense_block.abs()
        for stat in stats:
            stat.update(abs_block)


@dataclass
class DenseLoraMagnitudeAggregator:
    scaling: float
    out_chunk_size: int = 8
    overall: RunningMagnitudeStats = field(default_factory=RunningMagnitudeStats)
    by_module: dict[str, RunningMagnitudeStats] = field(default_factory=dict)
    num_examples: int = 0
    num_batches: int = 0
    num_layer_matrices: int = 0

    def reset(self) -> None:
        self.overall = RunningMagnitudeStats()
        self.by_module = {}
        self.num_examples = 0
        self.num_batches = 0
        self.num_layer_matrices = 0

    def update(self, combined_loras: dict[str, dict[str, Tensor]]) -> None:
        if not combined_loras:
            return

        first_module = next(iter(combined_loras.values()))
        first_A = first_module["A"]
        batch_size, num_layers = first_A.shape[:2]
        self.num_examples += int(batch_size)
        self.num_batches += 1
        self.num_layer_matrices += int(batch_size * num_layers)

        for module_name, module_lora in combined_loras.items():
            module_stats = self.by_module.setdefault(
                module_name, RunningMagnitudeStats()
            )
            _update_dense_magnitude_stats_for_module(
                [module_stats, self.overall],
                module_lora["A"],
                module_lora["B"],
                scaling=self.scaling,
                out_chunk_size=self.out_chunk_size,
            )

    def to_dict(self) -> dict[str, object]:
        return {
            "scaling": self.scaling,
            "out_chunk_size": self.out_chunk_size,
            "num_examples": self.num_examples,
            "num_batches": self.num_batches,
            "num_layer_matrices": self.num_layer_matrices,
            "overall": self.overall.to_dict(),
            "by_module": {
                module_name: stats.to_dict()
                for module_name, stats in sorted(self.by_module.items())
            },
        }
