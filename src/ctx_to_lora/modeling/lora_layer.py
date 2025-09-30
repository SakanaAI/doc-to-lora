from collections.abc import Iterable
from functools import partial
from operator import attrgetter

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor

from ctx_to_lora.utils import get_layers


def lora_forward(
    x: Float[Tensor, "tot_q seq_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "tot_q seq_len d_out"]:
    # A: [n_ctx, r, d_in] -> [tot_q, r, d_in]
    A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    # B: [n_ctx, d_out, r] -> [tot_q, d_out, r]
    B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)

    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    delta_x = einsum(A, delta_x, "tot_q r d_in, tot_q s_len d_in -> tot_q s_len r")
    delta_x = einsum(B, delta_x, "tot_q r d_out, tot_q s_len r -> tot_q s_len d_out")
    delta_x = delta_x * scaling
    return (base_out + delta_x).to(base_out.dtype)


def lora_forward_packed(
    x: Float[Tensor, "1 tot_len d_in"],
    n_qs: Integer[Tensor, "n_ctx"],
    tot_q: int,
    seq_lens: Integer[Tensor, "tot_q"],
    tot_len: int,
    A: Float[Tensor, "n_ctx r d_in"],
    B: Float[Tensor, "n_ctx r d_out"],
    lora_dropout_p: float,
    scaling: float,
    self,
    *args,
    **kwargs,
) -> Float[Tensor, "1 tot_len d_out"]:
    # bs of x should be 1 in this case
    base_out = torch.nn.Linear.forward(self, x, *args, **kwargs)
    x = x.to(A.dtype)
    delta_x = F.dropout(x, p=lora_dropout_p, training=self.training)
    repeated_A = A.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_A = repeated_A.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    repeated_B = B.repeat_interleave(n_qs, dim=0, output_size=tot_q)
    repeated_B = repeated_B.repeat_interleave(seq_lens, dim=0, output_size=tot_len)

    delta_x = einsum(
        repeated_A, delta_x, "tot_len r d_in, bs tot_len d_in -> bs tot_len r"
    )
    delta_x = einsum(
        repeated_B, delta_x, "tot_len r d_out, bs tot_len r -> bs tot_len d_out"
    )
    delta_x = delta_x * scaling

    return (base_out + delta_x).to(base_out.dtype)


def apply_lora_to_layers(
    model: torch.nn.Module,
    layer_indices: Iterable[int],
    generated_loras: dict[str, dict[str, Float[Tensor, "n_ctx n_layers r _"]]],
    n_qs: Integer[Tensor, "n_ctx"],
    position_ids: Integer[Tensor, "bs seq_len"] = None,
) -> None:
    layers = get_layers(model)
    if position_ids is not None:
        position_ids = position_ids.squeeze(0)
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1
        tot_len = seq_lens.sum().item()
    tot_q = n_qs.sum().item()
    for layer_idx in layer_indices:
        layer = layers[layer_idx]

        for mname in generated_loras:
            if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
                long_mname = f"self_attn.{mname}"
            elif mname in ["down_proj", "up_proj", "gate_proj"]:
                long_mname = f"mlp.{mname}"
            module = attrgetter(long_mname)(layer)
            A = generated_loras[mname]["A"][:, layer_idx]
            B = generated_loras[mname]["B"][:, layer_idx]
            module.forward = partial(module.forward, n_qs=n_qs, tot_q=tot_q, A=A, B=B)
            if position_ids is not None:
                module.forward = partial(
                    module.forward, seq_lens=seq_lens, tot_len=tot_len
                )
