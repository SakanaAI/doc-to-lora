"""
Utilities for merging / aggregating LoRA adapters coming from multiple chunks.
"""

from typing import Literal

import torch
from einops import rearrange
from jaxtyping import Integer
from torch import Tensor


def compute_rank(n_lora, rank, aggregation):
    """Return the effective (max) rank required for a group.

    aggregation == 'mean': we only concatenate the existing LoRA ranks (n_lora * r)
    aggregation == 'sum' : an additional bias term may be appended (+1 * r)
    """
    if aggregation == "mean":
        return n_lora * rank
    elif aggregation == "sum":
        return (n_lora + 1) * rank
    else:
        raise ValueError(f"Unknown aggregation: {aggregation}")


def combine_lora(
    generated_loras: dict[str, dict[str, Tensor]],
    n_chunks: Integer[Tensor, "n_ctx"],
    aggregation: Literal["mean", "sum"],
    lora_bias: dict[str, dict[str, Tensor]] = None,
) -> dict[str, dict[str, Tensor]]:
    """Combine per-chunk LoRA adapter weights into grouped higher-rank adapters.

    Parameters
    ----------
    generated_loras
        Mapping: module_name -> {'A': Tensor, 'B': Tensor}
        Each tensor shape: [total_chunks, n_layers, r, dim] for A, and
        [total_chunks, n_layers, dim, r] for B (but we treat generically here).
    n_chunks
        1D tensor giving number of chunks in each aggregation group. len(n_chunks)=G.
    aggregation
        'mean' or 'sum'. Controls scaling and bias handling.
    lora_bias
        Optional mapping mirroring generated_loras providing bias LoRA tensors (same
        per-chunk shape except missing the leading chunk dimension which is expanded).

    Returns
    -------
    dict
        module_name -> {'A': combined_A, 'B': combined_B}
        combined_A shape: [G, n_layers, max_rank, dim]
        combined_B shape: [G, n_layers, dim, max_rank]
    """
    assert aggregation in ["mean", "sum"]

    # Precompute common quantities
    num_groups = len(n_chunks)
    total_chunks = int(n_chunks.sum())
    sqrt_group_sizes = n_chunks**0.5  # [G]
    # Repeat for each chunk within a group: e.g.,
    # if n_chunks=[2,3] -> repeat factors [sqrt2, sqrt2, sqrt3, sqrt3, sqrt3]
    sqrt_group_sizes_per_chunk = sqrt_group_sizes.repeat_interleave(n_chunks, dim=0)

    # Assume all modules share same base rank r
    # (second-to-last dim index for A, last dim for B)
    # Use first module's A tensor to infer rank
    first_module = next(iter(generated_loras))
    base_rank = generated_loras[first_module]["A"].shape[-2]
    max_rank_needed = max(compute_rank(n, base_rank, aggregation) for n in n_chunks)

    # Initialize output container
    combined_loras: dict[str, dict[str, Tensor]] = {
        module: {"A": None, "B": None} for module in generated_loras.keys()
    }

    # Iterate over modules and LoRA matrices A / B.
    # For A we concatenate along its rank dimension index=2; for B along index=3.
    for module_name, module_loras in generated_loras.items():
        for matrix_key, rank_dim in (("A", 2), ("B", 3)):
            # Shape conventions (both stored with leading chunk axis):
            #   loras: [total_chunks, n_layers, r, dim] (A)
            #       or [total_chunks, n_layers, dim, r] (B)
            loras = module_loras[matrix_key]

            # Expand bias if provided
            bias_tensor = None
            if lora_bias is not None:
                bias_tensor = lora_bias[module_name][matrix_key].expand(
                    num_groups, *loras.shape[1:]
                )

            # For 'mean' aggregation, scale each chunk's LoRA by sqrt(group_size)
            if aggregation == "mean":
                loras = loras / sqrt_group_sizes_per_chunk.view(total_chunks, 1, 1, 1)

            # Split the leading chunk dimension into groups per n_chunks specification
            per_group_list = loras.split(n_chunks.tolist(), dim=0)

            # Concat each group
            if matrix_key == "A":
                rearrange_pattern = "chunks n_layers r dim -> 1 n_layers (chunks r) dim"
            else:
                rearrange_pattern = "chunks n_layers dim r -> 1 n_layers dim (chunks r)"
            per_group_deltas = [
                rearrange(group_tensor, rearrange_pattern)
                for group_tensor in per_group_list
            ]

            # Delegate to helper for combining rank expansion across groups.
            combined_loras[module_name][matrix_key] = _combine_single_matrix(
                per_group_deltas=per_group_deltas,
                bias=bias_tensor,
                n_chunks=n_chunks,
                base_rank=base_rank,
                max_rank_needed=max_rank_needed,
                aggregation=aggregation,
                matrix_key=matrix_key,
                rank_dim=rank_dim,
            )

    return combined_loras


def _combine_single_matrix(
    *,
    per_group_deltas: list[Tensor],
    bias: Tensor | None,
    n_chunks: Tensor,
    base_rank: int,
    max_rank_needed: int,
    aggregation: str,
    matrix_key: str,
    rank_dim: int,
) -> Tensor:
    """Assemble a single LoRA matrix (either 'A' or 'B') across groups.

    Parameters
    ----------
    loras
        Tensor of shape (total_chunks, n_layers, r, dim) if 'A' else
        (total_chunks, n_layers, dim, r) if 'B'.
    per_group_deltas
        A list of concat'd tensors
    bias_tensor
        Expanded bias tensor of shape (G, n_layers, r, dim) for 'A' or
        (G, n_layers, dim, r) for 'B', or None.
    n_chunks
        1D tensor (G,) with number of chunks per group.
    base_rank
        Original (per-LoRA) rank r.
    max_rank_needed
        Maximum rank capacity allocated for any group (accounts for bias when summing).
    aggregation
        'mean' or 'sum'. For 'sum' we optionally append a bias-related term.
    matrix_key
        'A' or 'B'. Determines which axis is the rank axis and bias handling sign.
    rank_dim
        Dimension along which to concatenate within each group's tensors.

    Returns
    -------
    Tensor
        Combined LoRA matrix with shape (G, n_layers, max_rank, dim) for 'A'
        or (G, n_layers, dim, max_rank) for 'B'.
    """
    assert matrix_key in ("A", "B"), f"matrix_key must be 'A' or 'B', got {matrix_key}"

    num_groups = len(n_chunks)
    combined_shape = list(bias.shape)
    combined_shape[0] = num_groups

    # Adjust the rank dimension size placeholder.
    if matrix_key == "A":
        rank_dim_index = -2  # (G, layers, rank, dim)
    else:
        rank_dim_index = -1  # (G, layers, dim, rank)
    combined_shape[rank_dim_index] = max_rank_needed

    combined = torch.zeros(*combined_shape, device=bias.device, dtype=bias.dtype)

    for g, deltas in enumerate(per_group_deltas):
        # we remove leading dim when rearrange
        combined_rank = deltas.shape[rank_dim]

        # Build slice pattern: rank slice up to combined_rank.
        slice_pattern = [g, slice(None), slice(None), slice(None)]
        slice_pattern[rank_dim] = slice(combined_rank)
        if bias is not None and aggregation == "sum":
            if matrix_key == "A":
                deltas = rearrange(
                    deltas,
                    "1 n_layers (chunks r) dim -> chunks n_layers r dim",
                    chunks=n_chunks[g],
                )
            else:
                deltas = rearrange(
                    deltas,
                    "1 n_layers dim (chunks r) -> chunks n_layers dim r",
                    chunks=n_chunks[g],
                )

            center = deltas.mean(dim=0, keepdim=True)
            deltas = deltas - center

            if matrix_key == "A":
                deltas = rearrange(
                    deltas, "chunks n_layers r dim -> 1 n_layers (chunks r) dim"
                )
                cat_lora = torch.cat(
                    [deltas, center],
                    dim=rank_dim,
                )
                combined[g, :, : (1 + n_chunks[g]) * base_rank, :] = cat_lora
            else:
                deltas = rearrange(
                    deltas, "chunks n_layers dim r -> 1 n_layers dim (chunks r)"
                )
                combined[g, :, :, : (1 + n_chunks[g]) * base_rank] = torch.cat(
                    [deltas, center],
                    dim=rank_dim,
                )

        else:
            combined[slice_pattern] = deltas

    return combined
