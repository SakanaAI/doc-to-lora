from functools import partial

from liger_kernel.transformers import monkey_patch
from liger_kernel.transformers.functional import liger_cross_entropy
from liger_kernel.transformers.geglu import LigerGEGLUMLP
from liger_kernel.transformers.monkey_patch import (
    _bind_method_to_module,
    _patch_rms_norm_module,
)
from liger_kernel.transformers.rope import liger_rotary_pos_emb
from transformers import PreTrainedModel


def apply_liger_kernel_to_gemma3_text_patched(
    rope: bool = True,
    cross_entropy: bool = False,
    fused_linear_cross_entropy: bool = True,
    rms_norm: bool = True,
    geglu: bool = True,
    model: PreTrainedModel = None,
) -> None:
    """
    Apply Liger kernels to replace original implementation in HuggingFace Gemma3

    Args:
        rope (bool): Whether to apply Liger's rotary position embedding. Default is True.
        cross_entropy (bool): Whether to apply Liger's cross entropy loss. Default is False.
        fused_linear_cross_entropy (bool):
            Whether to apply Liger's fused linear cross entropy loss. Default is True.
            `cross_entropy` and `fused_linear_cross_entropy` cannot both be True.
            If `fused_linear_cross_entropy` is True, the logits will not be materialized but more memory efficient.
        rms_norm (bool): Whether to apply Liger's RMSNorm. Default is True.
        geglu (bool): Whether to apply Liger's GeGLU MLP. Default is True.
        model (PreTrainedModel): The model instance to apply Liger kernels to, if the model has already been
        loaded. Default is None.
    """
    assert not (cross_entropy and fused_linear_cross_entropy), (
        "cross_entropy and fused_linear_cross_entropy cannot both be True."
    )

    ###
    from liger_kernel.transformers.gema3_rms import LigerRMSNormForGemma3
    from liger_kernel.transformers.model.gemma3 import causal_forward
    from transformers.models.gemma3 import modeling_gemma3

    ### PATCHED
    from transformers.models.gemma3.modeling_gemma3 import (
        Gemma3DecoderLayer,
        Gemma3ForCausalLM,
        Gemma3TextModel,
    )

    _patch_rms_norm_module_for_gemma3 = partial(
        _patch_rms_norm_module, offset=1.0, casting_mode="gemma", in_place=False
    )

    if rope:
        modeling_gemma3.apply_rotary_pos_emb = liger_rotary_pos_emb

    if rms_norm:
        modeling_gemma3.Gemma3RMSNorm = LigerRMSNormForGemma3

    if geglu:
        modeling_gemma3.Gemma3MLP = LigerGEGLUMLP

    # Handle loss function
    if cross_entropy:
        from transformers.loss.loss_utils import nn

        nn.functional.cross_entropy = liger_cross_entropy

    if fused_linear_cross_entropy:
        modeling_gemma3.Gemma3ForCausalLM.forward = causal_forward

    if model is not None:
        # The model instance already exists, so we need to additionally patch the
        # instance variables that reference already-instantiated modules

        ### PATCHED
        if isinstance(model, Gemma3ForCausalLM) or isinstance(model, Gemma3TextModel):
            # get the base model from the model instance
            base_model = model.model if isinstance(model, Gemma3ForCausalLM) else model
            ###

            if rms_norm:
                _patch_rms_norm_module_for_gemma3(base_model.norm)

            for decoder_layer in base_model.layers:
                decoder_layer: Gemma3DecoderLayer
                if geglu:
                    _bind_method_to_module(
                        decoder_layer.mlp, "forward", LigerGEGLUMLP.forward
                    )
                if rms_norm:
                    _patch_rms_norm_module_for_gemma3(decoder_layer.input_layernorm)
                    _patch_rms_norm_module_for_gemma3(
                        decoder_layer.post_attention_layernorm
                    )
                    _patch_rms_norm_module_for_gemma3(
                        decoder_layer.pre_feedforward_layernorm
                    )
                    _patch_rms_norm_module_for_gemma3(
                        decoder_layer.post_feedforward_layernorm
                    )
                    _patch_rms_norm_module_for_gemma3(decoder_layer.self_attn.q_norm)
                    _patch_rms_norm_module_for_gemma3(decoder_layer.self_attn.k_norm)

        else:
            raise TypeError("The model must be Gemma3ForCausalLM.")


monkey_patch.apply_liger_kernel_to_gemma3_text = (
    apply_liger_kernel_to_gemma3_text_patched
)
monkey_patch.MODEL_TYPE_TO_APPLY_LIGER_FN["gemma3_text"] = (
    apply_liger_kernel_to_gemma3_text_patched
)
