import logging
from collections.abc import Callable, Iterable
from operator import attrgetter

import torch
import torch.nn.functional as F
from einops import einsum
from jaxtyping import Float, Integer
from torch import Tensor
from torch.utils.hooks import RemovableHandle

from ctx_to_lora.utils import get_layers

logger = logging.getLogger()


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
    pre_hook: Callable | None = None,
    post_hook: Callable | None = None,
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
    pre_hook: Callable | None = None,
    post_hook: Callable | None = None,
) -> tuple[RemovableHandle | None, RemovableHandle | None]:
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
    pre_hook: Callable | None = None,
    post_hook: Callable | None = None,
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
    position_ids: Integer[Tensor, "bs seq_len"] | None = None,
    training: bool = False,
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

    if position_ids is not None:
        position_ids = position_ids.squeeze()
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1

    def lora_hook_packed_sequence(
        module: torch.nn.Module,
        args: tuple | Float[Tensor, "bs seq_len d_in"],
        output: Float[Tensor, "bs seq_len d_out"],
    ) -> Float[Tensor, "bs seq_len d_out"]:
        if isinstance(output, tuple):
            model_out = output[0]
        else:
            model_out = output

        # bs of x should be 1 in this case
        x = args[0].to(A.dtype)  # [1, tot_seq_len, d_in]
        # print(f"seq_lens: {seq_lens}")
        # print(f"sum of seq_lens: {seq_lens.sum()}")
        # print(f"tot_len: {x.shape[1]}")
        delta_x = F.dropout(x, input_dropout, training)
        n_seq = A.shape[0]
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
        newoutput = model_out + delta_x.to(model_out.dtype)
        if isinstance(output, tuple):
            return (newoutput, *output[1:])
        else:
            return newoutput

    hook_fn = lora_hook if position_ids is None else lora_hook_packed_sequence

    return apply_hook_to_layers(model, [module_name], [layer_index], post_hook=hook_fn)


def add_generated_layernorm_hook(
    model: torch.nn.Module,
    module_name: str,
    layer_index: int,
    W: Float[Tensor, "bs hidden_size"],
    training: bool,
    position_ids: Integer[Tensor, "bs seq_len"] | None = None,
) -> list[RemovableHandle]:
    """
    Adds layer normalization hooks to specified modules and layers.

    Args:
        model (torch.nn.Module): Model to hook
        module_name (str): Name of layernorm module (e.g., "input_layernorm")
        layer_index (int): Index of layer to modify
        W (Tensor): Learned weight tensor of shape [batch_size, hidden_size]
        training (bool): Whether model is in training mode
        position_ids (Optional[Tensor]): Position IDs for packed sequence processing

    Returns:
        list[RemovableHandle]: Hook handles for removal
    """

    def layernorm_hook(
        module: torch.nn.Module,
        args: tuple,
        output: Float[Tensor, "bs seq_len hidden_size"],
    ) -> Float[Tensor, "bs seq_len hidden_size"]:
        if isinstance(output, tuple):
            main_output = output[0]
            rest = output[1:]
        else:
            main_output = output
            rest = None

        x = args[0].to(W.dtype)
        # Apply learned weights to layernorm output
        # Unsqueeze to add seq_len dimension for broadcasting
        scaled_output = x * W.unsqueeze(1)
        new_output = main_output + scaled_output.to(output.dtype)

        return (new_output, *rest) if rest else new_output

    if position_ids is not None:
        position_ids = position_ids.squeeze()
        seq_lens = position_ids[torch.where(position_ids == 0)[0][1:] - 1]
        seq_lens = torch.cat(
            [seq_lens, torch.tensor([position_ids[-1]], device=seq_lens.device)]
        )
        seq_lens += 1

    def layernorm_hook_packed_sequence(
        module: torch.nn.Module,
        args: tuple,
        output: Float[Tensor, "bs seq_len hidden_size"],
    ) -> Float[Tensor, "bs seq_len hidden_size"]:
        if isinstance(output, tuple):
            main_output = output[0]
            rest = output[1:]
        else:
            main_output = output
            rest = None

        # [1, tot_seq_len, d_in]
        x = args[0].to(W.dtype)
        # Repeat weights for each sequence

        repeated_W = W.repeat_interleave(seq_lens, dim=0, output_size=seq_lens.sum())
        # Apply learned weights to layernorm output
        scaled_output = x * repeated_W.unsqueeze(0)
        new_output = main_output + scaled_output.to(output.dtype)

        return (new_output, *rest) if rest else new_output

    hook_fn = layernorm_hook if position_ids is None else layernorm_hook_packed_sequence

    return apply_hook_to_layers(model, [module_name], [layer_index], post_hook=hook_fn)
