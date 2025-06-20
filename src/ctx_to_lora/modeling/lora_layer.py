from collections.abc import Iterable
from functools import partial
from operator import attrgetter

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ctx_to_lora.utils import get_layers

# Patch corresponding layers of LLMs to include A and B lora matrices
"""
def forward(self, x: torch.Tensor, *args: Any, **kwargs: Any) -> torch.Tensor:
    self._check_forward_args(x, *args, **kwargs)
    adapter_names = kwargs.pop("adapter_names", None)

    if self.disable_adapters:
        if self.merged:
            self.unmerge()
        result = self.base_layer(x, *args, **kwargs)
    elif adapter_names is not None:
        result = self._mixed_batch_forward(x, *args, adapter_names=adapter_names, **kwargs)
    elif self.merged:
        result = self.base_layer(x, *args, **kwargs)
    else:
        result = self.base_layer(x, *args, **kwargs)
        torch_result_dtype = result.dtype

        lora_A_keys = self.lora_A.keys()
        for active_adapter in self.active_adapters:
            if active_adapter not in lora_A_keys:
                continue

            lora_A = self.lora_A[active_adapter]
            lora_B = self.lora_B[active_adapter]
            dropout = self.lora_dropout[active_adapter]
            scaling = self.scaling[active_adapter]
            x = self._cast_input_dtype(x, lora_A.weight.dtype)

            if not self.use_dora[active_adapter]:
                result = result + lora_B(lora_A(dropout(x))) * scaling
            else:
                if isinstance(dropout, nn.Identity) or not self.training:
                    base_result = result
                else:
                    x = dropout(x)
                    base_result = None

                result = result + self.lora_magnitude_vector[active_adapter](
                    x,
                    lora_A=lora_A,
                    lora_B=lora_B,
                    scaling=scaling,
                    base_layer=self.get_base_layer(),
                    base_result=base_result,
                )

        result = result.to(torch_result_dtype)

    return result
"""


def lora_forward(
    x: Float[Tensor, "bs seq_len d_in"],
    self,
    # pre_modulated_forward_fn: callable[..., Float[Tensor, "bs seq_len d_out"]],
    A: Float[Tensor, "bs r d_in"],
    B: Float[Tensor, "bs d_out r"],
    lora_dropout_p: float,
    scaling: float,
    *args,
    **kwargs,
) -> Float[Tensor, "bs seq_len d_out"]:
    """
    Forward pass of the LoRA layer.
    # TODO: implement the packed version
    # TODO: implement the one-lora-multi-query version

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, input_dim).
        A (torch.Tensor): LoRA matrix A with shape (bs, r, d_in).
        B (torch.Tensor): LoRA matrix B with shape (bs, d_out, r).

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, output_dim).
    """
    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "bs r d_in, bs seq_len d_in -> bs seq_len r")
    delta_x = einsum(B, delta_x, "bs d_out r, bs seq_len r -> bs seq_len d_out")
    delta_x = delta_x * scaling
    return base_out + delta_x


def lora_forward_packed(
    x: Float[Tensor, "bs seq_len d_in"],
    seq_lens: Integer[Tensor, "bs"],
    self,
    A: Float[Tensor, "bs r d_in"],
    B: Float[Tensor, "bs d_out r"],
    lora_dropout_p: float,
    scaling: float,
    *args,
    **kwargs,
) -> Float[Tensor, "bs seq_len d_out"]:
    """
    Forward pass of the LoRA layer for packed inputs.

    Args:
        x (torch.Tensor): Input tensor with shape (batch_size, input_dim).
        A (torch.Tensor): LoRA matrix A with shape (input_dim, lora_dim).
        B (torch.Tensor): LoRA matrix B with shape (lora_dim, output_dim).

    Returns:
        torch.Tensor: Output tensor with shape (batch_size, output_dim).
    """
    # bs of x should be 1 in this case
    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)

    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    # [tot_seq_len, r, d_in]
    repeated_A = A.repeat_interleave(seq_lens, dim=0, output_size=seq_lens.sum())
    # [tot_seq_len, d_out, r]
    repeated_B = B.repeat_interleave(seq_lens, dim=0, output_size=seq_lens.sum())
    delta_x = einsum(
        repeated_A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r"
    )
    delta_x = einsum(
        repeated_B, delta_x, "tot_len d_out r, bs tot_len r -> bs tot_len d_out"
    )
    delta_x = delta_x * scaling

    return base_out + delta_x


def apply_lora_to_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
    generated_loras: dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]],
    position_ids: Integer[Tensor, "bs seq_len"] = None,
) -> None:
    layers = get_layers(model)
    if position_ids is not None:
        position_ids = position_ids.squeeze()
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1
    for layer_idx in layer_indices:
        layer = layers[layer_idx]

        for mname in generated_loras:
            if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
                long_mname = f"self_attn.{mname}"
            elif mname in ["down_proj", "up_proj", "gate_proj"]:
                long_mname = f"mlp.{mname}"
            module = attrgetter(long_mname)(layer)
            module.forward = partial(
                module.forward,
                A=generated_loras[mname]["A"][:, layer_idx],
                B=generated_loras[mname]["B"][:, layer_idx],
            )
            if position_ids is not None:
                module.forward = partial(
                    module.forward,
                    seq_lens=seq_lens,
                )
