import logging
from operator import attrgetter
from typing import Callable, Iterable, Optional

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float
from torch import Tensor
from torch.utils.hooks import RemovableHandle
from utils import get_layers

logger = logging.getLogger(__name__)


def remove_hook_handles(handles: list[RemovableHandle]) -> None:
    """
    Removes all hook handles to detach them from the model.

    Args:
        handles (list[RemovableHandle]): A list of hook handles to be removed.
    """
    for handle in handles:
        if handle is not None:
            handle.remove()


def apply_hook_to_model(
    model: torch.nn.Module,
    module_name: str,
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
) -> torch.nn.Module:
    if not ((pre_hook is not None) or (post_hook is not None)):
        raise ValueError("No hooks provided. Nothing to apply.")

    module = attrgetter(module_name)(model)

    if pre_hook is not None:
        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
    if post_hook is not None:
        post_hook_handle = module.register_forward_hook(post_hook)
    return pre_hook_handle, post_hook_handle


def apply_hook_to_layer(
    layer: torch.nn.Module,
    mname: str,
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
) -> tuple[Optional[RemovableHandle], Optional[RemovableHandle]]:
    """
    Applies pre and/or post hooks to a specific layer in the model.

    Args:
        layer (torch.nn.Module): The layer to which hooks will be applied.
        mname (str): The name of the module within the layer.
        pre_hook (Optional[Callable], optional): Function to be called before the forward pass.
        post_hook (Optional[Callable], optional): Function to be called after the forward pass.

    Returns:
        Tuple[Optional[RemovableHandle], Optional[RemovableHandle]]:
            Handles for the pre and post hooks.
    """
    if not ((pre_hook is not None) or (post_hook is not None)):
        raise ValueError("No hooks provided. Nothing to apply.")

    pre_hook_handle = None
    post_hook_handle = None

    if mname in ["q_proj", "k_proj", "v_proj", "o_proj", "qkv_proj"]:
        mname = f"self_attn.{mname}"
    elif mname in ["down_proj", "up_proj", "gate_proj"]:
        mname = f"mlp.{mname}"
    module = attrgetter(mname)(layer)

    if pre_hook is not None:
        pre_hook_handle = module.register_forward_pre_hook(pre_hook)
    if post_hook is not None:
        post_hook_handle = module.register_forward_hook(post_hook)
    return pre_hook_handle, post_hook_handle


def apply_hook_to_layers(
    model: torch.nn.Module,
    module_names: list[str],
    layer_indices: Iterable[int],
    pre_hook: Optional[Callable] = None,
    post_hook: Optional[Callable] = None,
) -> list[RemovableHandle]:
    """
    Applies custom hooks to specified layers and modules in the model.

    Args:
        model (torch.nn.Module): The model to which hooks will be applied.
        module_names (list[str]): Names of the modules to hook.
        layer_indices (Iterable[int]): Indices of the layers to hook.
        pre_hook (Optional[Callable], optional): Function to be called before the forward pass.
        post_hook (Optional[Callable], optional): Function to be called after the forward pass.

    Returns:
        list[RemovableHandle]: A list of hook handles.
    """

    layers = get_layers(model)
    out_handles = []
    c = 0
    for layer_idx in layer_indices:
        layer = layers[layer_idx]
        for mname in module_names:
            handles = apply_hook_to_layer(
                layer,
                mname,
                pre_hook=pre_hook,
                post_hook=post_hook,
            )
            out_handles += handles
            if handles[0] is not None or handles[1] is not None:
                c += 1
    if c == 0:
        raise ValueError(
            "No forward hooks applied. Something might be wrong. "
            "Check if the module names are correct."
        )
    return out_handles


def add_generated_lora_hook(
    model: torch.nn.Module,
    module_name: str,
    layer_index: int,
    A: Float[Tensor, "bs r d_in"],
    B: Float[Tensor, "bs d_out r"],
    scaling: float,
    input_dropout: float,
    training: bool,
) -> list[RemovableHandle]:
    """
    Adds LoRA hooks to the specified modules and layers of the model.

    Args:
        model (torch.nn.Module): The model to which hooks will be applied.
        module_name (str): Name of the module to hook.
        A (Tensor): Weight tensor A.
        B (Tensor): Weight tensor B.
        scaling (float): Scaling factor.
        input_dropout (float): Dropout rate for input.
        training (bool): Flag indicating if the model is in training mode.

    Returns:
        list[RemovableHandle]: A list of hook handles.
    """

    def lora_hook(
        module: torch.nn.Module,
        args: tuple | Float[Tensor, "bs seq_len d_in"],
        output: Float[Tensor, "bs seq_len d_out"],
    ) -> Float[Tensor, "bs seq_len d_out"]:

        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        x = args[0].to(A.dtype)  # [bs, seq_len, d_in]

        delta_x = F.dropout(x, input_dropout, training)
        delta_x = einsum(A, delta_x, "bs r d_in, bs seq_len d_in -> bs seq_len r")
        delta_x = einsum(B, delta_x, "bs d_out r, bs seq_len r -> bs seq_len d_out")
        delta_x = delta_x * scaling

        # # A and B repeat for each input token
        # lora_A = A.repeat_interleave(seq_len, dim=0)
        # lora_B = B.repeat_interleave(seq_len, dim=0)
        # x = x.reshape(bs * seq_len, 1, -1)
        # delta_x = (
        #     torch.bmm(torch.bmm(F.dropout(x, input_dropout, training), lora_A), lora_B)
        #     * scaling
        # )

        newoutput = model_out + delta_x.to(model_out.dtype)
        if isinstance(output, tuple):
            return (newoutput, *output[1:])
        else:
            return newoutput

    return apply_hook_to_layers(model, [module_name], [layer_index], post_hook=lora_hook)
