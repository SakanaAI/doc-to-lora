"""Test / diagnostic script for comparing current per_ctx_loss implementation
against a proposed fixed version that detects label spans via mask transitions.

Run:
  python tmp/test_per_ctx_loss.py

It will:
  * Construct synthetic batches with known label span structure.
  * Call the existing per_ctx_loss (imported from ctx_to_lora.trainer).
  * Call a fixed implementation (fixed_per_ctx_loss) using label_mask transitions.
  * Show detected spans, per-query losses, per-context losses, and scaling ratios.
  * Highlight cases where the current implementation fails (empty spans / mismatch).

Note: The current per_ctx_loss assumes the provided `loss` vector is ordered
exactly as the concatenation of all label-token losses (no unlabeled tokens),
matching DistillationTrainer's usage, but NOT CrossEntropyTrainer's raw CE output.
"""

from __future__ import annotations

import math

# Ensure "src" is on path so we can import project modules when running from project root
import sys
import traceback
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from ctx_to_lora.trainer import per_ctx_loss  # type: ignore


@dataclass
class SyntheticCase:
    name: str
    label_mask: list[bool]
    n_queries_per_ctx: list[int]
    description: str


def per_ctx_loss(inputs, labels, loss):
    n_queries_per_ctx = inputs["n_queries"].tolist()

    position_ids = inputs["position_ids"].squeeze(0)
    # account only label positions
    label_mask = labels.squeeze(0) != -100
    label_pos_ids = label_mask * position_ids
    label_pos_ids_diff = label_pos_ids.diff(
        append=torch.tensor([0], device=position_ids.device)
    )
    # assumes the input starts with non-assistant tokens
    start_label_pos = torch.where((label_pos_ids_diff > 0) * ~label_mask)[0]
    end_label_pos = torch.where((label_pos_ids_diff < 0) * label_mask)[0]

    label_seq_lens = end_label_pos - start_label_pos
    cu_label_seq_lens = torch.cumsum(label_seq_lens, dim=0)
    start_indices = torch.cat(
        (
            torch.tensor([0], device=cu_label_seq_lens.device),
            cu_label_seq_lens[:-1],
        )
    )

    # these stack and split can be optimized but let's keep it simple
    # mean across tokens of each q
    qa_losses = torch.stack(
        [loss[start:end].mean() for start, end in zip(start_indices, cu_label_seq_lens)]
    )

    # mean across queries of each ctx
    per_ctx_losses = [ql.mean() for ql in torch.split(qa_losses, n_queries_per_ctx)]

    # per-ctx loss
    loss = torch.stack(per_ctx_losses)
    return loss


def fixed_per_ctx_loss(inputs, labels, loss_vec: torch.Tensor) -> torch.Tensor:
    """Reference implementation using label_mask transitions.

    Args:
        inputs: dict with keys: n_queries, position_ids
        labels: [1, L]
        loss_vec: tensor of shape (#label_tokens,) in label-token order.
    Returns:
        Tensor of shape (#contexts,) containing mean loss per context.
    """
    n_queries_per_ctx = inputs["n_queries"].tolist()
    label_mask = labels.squeeze(0) != -100

    lm = label_mask.to(torch.int)
    if lm.numel() == 0:
        return torch.empty(0)

    if lm.numel() == 1:
        spans = [(0, 1)] if lm[0] == 1 else []
    else:
        transition = lm[1:] - lm[:-1]
        starts = (transition == 1).nonzero(as_tuple=False).flatten() + 1
        ends = (transition == -1).nonzero(as_tuple=False).flatten() + 1
        if lm[0] == 1:
            starts = torch.cat([torch.tensor([0], device=starts.device), starts])
        if lm[-1] == 1:
            ends = torch.cat([ends, torch.tensor([lm.numel()], device=ends.device)])
        assert starts.numel() == ends.numel()
        spans = list(zip(starts.tolist(), ends.tolist()))

    # Extract per-query mean losses (loss_vec is already label-token sequential).
    lengths = [e - s for s, e in spans]
    assert sum(lengths) == loss_vec.numel(), (
        "Mismatch: total span label tokens != loss vector length."
    )
    cu_lengths = torch.tensor(lengths).cumsum(0)
    starts_loss = torch.cat([torch.tensor([0]), cu_lengths[:-1]])
    qa_means = [loss_vec[a:b].mean() for a, b in zip(starts_loss, cu_lengths)]
    qa_means_t = torch.stack(qa_means)

    # Group into contexts
    split_q = torch.split(qa_means_t, n_queries_per_ctx)
    per_ctx = torch.stack([ctx.mean() for ctx in split_q])
    return per_ctx


def build_labels_from_mask(mask: Sequence[bool]) -> torch.Tensor:
    # Assign dummy token id 1 where True, -100 elsewhere.
    data = [1 if m else -100 for m in mask]
    return torch.tensor([data])  # shape [1, L]


def build_position_ids(L: int) -> torch.Tensor:
    return torch.arange(L).unsqueeze(0)


def make_loss_vector(mask: Sequence[bool]) -> torch.Tensor:
    # Provide a smoothly varying loss value per label token to visualize averaging.
    # Example: losses increase linearly → weighting differences are obvious.
    values = []
    counter = 0
    for m in mask:
        if m:
            # synthetic harder tokens later
            values.append(0.5 + 0.1 * counter)
            counter += 1
    return torch.tensor(values)


def run_case(case: SyntheticCase):
    print(f"\n=== Case: {case.name} ===")
    print(case.description)
    mask = case.label_mask
    labels = build_labels_from_mask(mask)
    position_ids = build_position_ids(len(mask))
    loss_vec = make_loss_vector(mask)
    inputs = {
        "n_queries": torch.tensor(case.n_queries_per_ctx),
        "position_ids": position_ids,
    }
    print(f"Label mask: {''.join('1' if m else '.' for m in mask)}")
    print(f"Loss vector (per label token order): {loss_vec.tolist()}")

    # Attempt current implementation
    try:
        current = per_ctx_loss(inputs, labels, loss_vec)
        print(
            "Current per_ctx_loss output:",
            current.tolist(),
            "shape=",
            list(current.shape),
        )
    except Exception as e:
        print("Current per_ctx_loss raised exception:")
        traceback.print_exc()
        current = None

    # Fixed implementation
    try:
        fixed = fixed_per_ctx_loss(inputs, labels, loss_vec)
        print(
            "Fixed per_ctx_loss output:  ", fixed.tolist(), "shape=", list(fixed.shape)
        )
    except Exception as e:
        print("Fixed implementation raised exception:")
        traceback.print_exc()
        fixed = None

    # Compare magnitudes if both succeeded
    if current is not None and fixed is not None and current.numel() == fixed.numel():
        ratio = (
            (current.mean() / fixed.mean()).item() if fixed.mean() != 0 else math.nan
        )
        print(f"Mean(current)/Mean(fixed) ratio: {ratio:.4f}")
    print("---")


def main():
    cases = [
        # SyntheticCase(
        #     name="Two spans, single context",
        #     label_mask=[False, False, True, True, True, False, False, True, True],
        #     n_queries_per_ctx=[2],
        #     description="Two answer/query spans of lengths 3 and 2 in one context.",
        # ),
        SyntheticCase(
            name="Single span",
            label_mask=[False, True, True, True, False],
            n_queries_per_ctx=[1],
            description="One continuous labeled span.",
        ),
        # SyntheticCase(
        #     name="Three contexts (2,1 queries)",
        #     label_mask=[False, True, True, False, True, False, True, True, True, False],
        #     n_queries_per_ctx=[1, 1, 1],
        #     description="Three separate single-query contexts.",
        # ),
        # SyntheticCase(
        #     name="Edge case: starts labeled",
        #     label_mask=[True, True, False, True, False],
        #     n_queries_per_ctx=[2],
        #     description="Sequence begins with a labeled span then another later.",
        # ),
        # SyntheticCase(
        #     name="Edge case: ends labeled",
        #     label_mask=[False, True, False, True, True],
        #     n_queries_per_ctx=[2],
        #     description="Sequence ends with a labeled span.",
        # ),
    ]

    for case in cases:
        run_case(case)

    print("\nDone. Inspect above for discrepancies.\n")


if __name__ == "__main__":
    main()
