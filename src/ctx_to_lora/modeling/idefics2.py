# Copyright 2024 the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Idefics2 model."""

from copy import deepcopy
import math

import torch
import torch.utils.checkpoint
from torch import nn
from transformers.activations import ACT2FN
from transformers.cache_utils import Cache
from transformers.configuration_utils import PretrainedConfig
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask
from transformers.modeling_utils import PreTrainedModel
from transformers.models.idefics2.configuration_idefics2 import Idefics2Config
from transformers.utils import (
    add_start_docstrings,
    is_flash_attn_2_available,
    is_flash_attn_greater_or_equal_2_10,
    logging,
)

if is_flash_attn_2_available():
    from transformers.modeling_flash_attention_utils import _flash_attention_forward

logger = logging.get_logger(__name__)


class Idefics2PerceiverConfig(PretrainedConfig):
    r"""
    Configuration objects inherit from [`PretrainedConfig`] and can be used to control the model outputs. Read the
    documentation from [`PretrainedConfig`] for more information.

    Args:
        hidden_act (`str` or `function`, *optional*, defaults to `"silu"`):
            The non-linear activation function (function or string) in the perceiver block.
        hidden_size (`int`, *optional*, defaults to 4096):
            Dimension of the hidden representations.
        rms_norm_eps (`float`, *optional*, defaults to 1e-06):
            The epsilon used by the rms normalization layers.
        resampler_n_latents (`int`, *optional*, defaults to 64):
            Number of latent embeddings to resample ("compress") the input sequence to (usually < 128).
        resampler_depth (`int`, *optional*, defaults to 3):
            Depth of the Perceiver Resampler (Transformer w/ cross attention). Should be shallow (<= 3).
        resampler_n_heads (`int`, *optional*, defaults to 16):
            Number of heads in each Transformer block (for multi-headed self-attention).
        resampler_head_dim (`int`, *optional*, defaults to 96):
            Dimensionality of each head projection in the Transformer block.
        num_key_value_heads (`int`, *optional*, defaults to 4):
            Number of key-value heads in the perceiver attention block.
        attention_dropout (`float`, *optional*, defaults to 0.0):
            The dropout ratio for the attention probabilities.
    """

    model_type = "idefics2_perceiver"

    def __init__(
        self,
        input_size: int,
        intermediate_size_factor: int = 1,
        hidden_act="silu",
        hidden_size=4096,
        rms_norm_eps=1e-06,
        resampler_n_latents=64,
        resampler_depth=3,
        resampler_n_heads=16,
        resampler_head_dim=96,
        num_key_value_heads=4,
        attention_dropout=0.0,
        is_cross_attn=True,
        concat_latents_context=False,
        **kwargs,
    ):
        # for mlp
        self.concat_latents_context = concat_latents_context
        self.is_cross_attn = is_cross_attn
        self.input_size = input_size
        self.intermediate_size_factor = intermediate_size_factor
        # for perceiver
        self.hidden_act = hidden_act
        self.hidden_size = hidden_size
        self.rms_norm_eps = rms_norm_eps
        self.resampler_n_latents = resampler_n_latents
        self.resampler_depth = resampler_depth
        self.resampler_n_heads = resampler_n_heads
        self.num_key_value_heads = num_key_value_heads
        self.resampler_head_dim = resampler_head_dim
        self.attention_dropout = attention_dropout
        if self.num_key_value_heads > self.resampler_n_heads:
            raise ValueError(
                f"num_key_value_heads={self.num_key_value_heads} must be less than or equal to"
                f" resampler_n_heads={self.resampler_n_heads}"
            )
        super().__init__(**kwargs)


class Idefics2MLP(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        output_size: int,
        hidden_act: str,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=False)
        self.down_proj = nn.Linear(intermediate_size, output_size, bias=False)
        self.act_fn = ACT2FN[hidden_act]

    def forward(self, x):
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


IDEFICS2_START_DOCSTRING = r"""
    This model inherits from [`PreTrainedModel`]. Check the superclass documentation for the generic methods the
    library implements for all its model (such as downloading or saving, resizing the input embeddings, pruning heads
    etc.)

    This model is also a PyTorch [torch.nn.Module](https://pytorch.org/docs/stable/nn.html#torch.nn.Module) subclass.
    Use it as a regular PyTorch Module and refer to the PyTorch documentation for all matter related to general usage
    and behavior.

    Parameters:
        config ([`Idefics2Config`] or [`Idefics2VisionConfig`]):
            Model configuration class with all the parameters of the model. Initializing with a config file does not
            load the weights associated with the model, only the configuration. Check out the
            [`~PreTrainedModel.from_pretrained`] method to load the model weights.
"""


@add_start_docstrings(
    "The bare Idefics2 Model outputting raw hidden-states without any specific head on top.",
    IDEFICS2_START_DOCSTRING,
)
class Idefics2PreTrainedModel(PreTrainedModel):
    config_class = Idefics2Config
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _no_split_modules = [
        "Idefics2VisionAttention",
        "Idefics2MLP",
        "Idefics2PerceiverLayer",
        "Idefics2DecoderLayer",
    ]
    _skip_keys_device_placement = "past_key_values"
    _supports_flash_attn_2 = True
    _supports_sdpa = True
    _supports_cache_class = True

    def _init_weights(self, module):
        std = (
            self.config.initializer_range
            if hasattr(self.config, "initializer_range")
            else 0.02
        )

        if hasattr(module, "class_embedding"):
            module.class_embedding.data.normal_(mean=0.0, std=std)

        if isinstance(module, (nn.Linear, nn.Conv2d)):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()


# Copied from transformers.models.llama.modeling_llama.repeat_kv
def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    """
    This is the equivalent of torch.repeat_interleave(x, dim=1, repeats=n_rep). The hidden states go from (batch,
    num_key_value_heads, seqlen, head_dim) to (batch, num_attention_heads, seqlen, head_dim)
    """
    batch, num_key_value_heads, slen, head_dim = hidden_states.shape
    if n_rep == 1:
        return hidden_states
    hidden_states = hidden_states[:, :, None, :, :].expand(
        batch, num_key_value_heads, n_rep, slen, head_dim
    )
    return hidden_states.reshape(batch, num_key_value_heads * n_rep, slen, head_dim)


# Copied from transformers.models.llama.modeling_llama.LlamaRMSNorm with Llama->Idefics2
class Idefics2RMSNorm(nn.Module):
    def __init__(self, hidden_size, eps=1e-6):
        """
        Idefics2RMSNorm is equivalent to T5LayerNorm
        """
        super().__init__()
        self.weight = nn.Parameter(torch.ones(hidden_size))
        self.variance_epsilon = eps

    def forward(self, hidden_states):
        input_dtype = hidden_states.dtype
        hidden_states = hidden_states.to(torch.float32)
        variance = hidden_states.pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.variance_epsilon)
        return self.weight * hidden_states.to(input_dtype)

    def extra_repr(self):
        return f"{tuple(self.weight.shape)}, eps={self.variance_epsilon}"


class Idefics2PerceiverAttention(nn.Module):
    def __init__(self, config, layer_idx: int | None = None) -> None:
        """Perceiver Cross-Attention Module --> let long-form inputs be `context`, resampled embeddings be `latents`"""
        super().__init__()
        self.config = config
        self.layer_idx = None
        self.hidden_size = config.hidden_size
        self.num_heads = config.resampler_n_heads
        self.head_dim = config.resampler_head_dim
        self.num_key_value_heads = config.num_key_value_heads
        self.num_key_value_groups = self.num_heads // self.num_key_value_heads
        self.attention_dropout = config.attention_dropout

        self.q_proj = nn.Linear(
            self.hidden_size, self.num_heads * self.head_dim, bias=False
        )
        self.k_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.v_proj = nn.Linear(
            self.hidden_size, self.num_key_value_heads * self.head_dim, bias=False
        )
        self.o_proj = nn.Linear(
            self.num_heads * self.head_dim, self.hidden_size, bias=False
        )

        self.is_causal = False

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        """
        Runs Perceiver Self-Attention, with special (context, latents) appended along the `seq` dimension!

        Args:
            latents (`torch.Tensor`): Tensor of shape [bsz, n_latents, embed_dim] representing fixed length latents to compress to.
            context (`torch.Tensor`): Tensor of shape [bsz, seq, embed_dim] representing long-form context to resample.
            attention_mask (`torch.Tensor`, *optional*): Tensor of shape [bsz, 1, seq, n_latents] representing attention mask.
            position_ids (`torch.LongTensor`, *optional*): Tensor of shape [bsz, seq] representing position indices of each input token.
            past_key_value (`Tuple[torch.Tensor]`, *optional*): Tuple of tensors containing cached key and value states.
            output_attentions (`bool`, *optional*, defaults to `False`): Whether to return attention weights.
            use_cache (`bool`, *optional*, defaults to `False`): Whether to use past_key_value for caching.
        """
        bsz, q_len, _ = latents.size()
        kv_seq_len = q_len + context.size()[1]

        hidden_states = torch.concat([context, latents], dim=-2)

        query_states = self.q_proj(latents)
        key_states = self.k_proj(hidden_states)
        value_states = self.v_proj(hidden_states)

        query_states = query_states.view(
            bsz, q_len, self.num_heads, self.head_dim
        ).transpose(1, 2)
        key_states = key_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            bsz, kv_seq_len, self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        past_key_value = getattr(self, "past_key_value", past_key_value)

        if past_key_value is not None:
            key_states, value_states = past_key_value.update(
                key_states, value_states, self.layer_idx
            )

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)

        attn_weights = torch.matmul(
            query_states, key_states.transpose(2, 3)
        ) / math.sqrt(self.head_dim)

        if attn_weights.size() != (bsz, self.num_heads, q_len, kv_seq_len):
            raise ValueError(
                f"Attention weights should be of size {(bsz, self.num_heads, q_len, kv_seq_len)}, but is"
                f" {attn_weights.size()}"
            )

        if attention_mask is not None:
            if attention_mask.size() != (bsz, 1, q_len, kv_seq_len):
                raise ValueError(
                    f"Attention mask should be of size {(bsz, 1, q_len, kv_seq_len)}, but is {attention_mask.size()}"
                )

            attn_weights = attn_weights + attention_mask

        # upcast attention to fp32
        attn_weights = nn.functional.softmax(
            attn_weights, dim=-1, dtype=torch.float32
        ).to(query_states.dtype)
        attn_output = torch.matmul(attn_weights, value_states)

        if attn_output.size() != (bsz, self.num_heads, q_len, self.head_dim):
            raise ValueError(
                f"`attn_output` should be of size {(bsz, self.num_heads, q_len, self.head_dim)}, but is"
                f" {attn_output.size()}"
            )

        attn_output = attn_output.transpose(1, 2).contiguous()
        attn_output = attn_output.reshape(bsz, q_len, self.num_heads * self.head_dim)

        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


# NO LONGER EXIST Copied from transformers.models.mistral.modeling_mistral.MistralFlashAttention2 with MistralAttention->Idefics2PerceiverAttention,MistralFlashAttention->Idefics2PerceiverFlashAttention,Mistral->Idefics2
# TODO cyril: modular
class Idefics2PerceiverFlashAttention2(Idefics2PerceiverAttention):
    """
    Idefics2 flash attention module. This module inherits from `Idefics2PerceiverAttention` as the weights of the module stays
    untouched. The only required change would be on the forward pass where it needs to correctly call the public API of
    flash attention and deal with padding tokens in case the input contains any of them.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        # TODO: Should be removed once Flash Attention for RoCm is bumped to 2.1.
        # flash_attn<2.1 generates top-left aligned causal mask, while what is needed here is bottom-right alignement, that was made default for flash_attn>=2.1. This attribute is used to handle this difference. Reference: https://github.com/Dao-AILab/flash-attention/releases/tag/v2.1.0.
        # Beware that with flash_attn<2.1, using q_seqlen != k_seqlen (except for the case q_seqlen == 1) produces a wrong mask (top-left).
        self._flash_attn_uses_top_left_mask = not is_flash_attn_greater_or_equal_2_10()

    # Ignore copy
    def forward(
        self,
        latents: torch.Tensor,
        is_cross_attn: bool,
        concat_latents_context: bool,
        context: torch.Tensor | None = None,
        attention_mask: torch.LongTensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: Cache | None = None,
        output_attentions: bool = False,
        use_cache: bool = False,
        **kwargs,
    ) -> tuple[torch.Tensor, torch.Tensor | None, tuple[torch.Tensor] | None]:
        bsz, q_len, _ = latents.size()
        # kv_seq_len = q_len + context.size()[1]
        # kv_seq_len = context.size()[1]
        query_states = self.q_proj(latents)
        if is_cross_attn:
            kv_inp = context
            if concat_latents_context:
                # Query, Key, Value Projections --> Note that in Flamingo, latents are *concatenated* with context prior to attn!
                # Note: This results in queries w/ `seq = n_latents`, and keys, values with `seq = len(context) + n_latents`
                if context.shape[0] != latents.shape[0]:
                    # context: [1, tot_len, dim]
                    # latents: [bs, n_latents, dim]
                    # slice context sample by sample and concat with latents
                    old_cu_seq_lens_k = kwargs.pop("old_cu_seq_lens_k")
                    cu_seq_lens_k = kwargs["cu_seq_lens_k"]
                    old_cur_len = 0
                    cur_len = 0
                    n_latents = latents.shape[1]
                    kv_inp = torch.empty(
                        1,
                        cu_seq_lens_k[-1],
                        latents.shape[2],
                        dtype=context.dtype,
                        device=context.device,
                    )
                    for i, (old_cu_len, cu_len) in enumerate(
                        zip(old_cu_seq_lens_k[1:], cu_seq_lens_k[1:])
                    ):
                        ctx_len = old_cu_len - old_cur_len
                        old_end_idx = cur_len + ctx_len
                        kv_inp[0, cur_len:old_end_idx] = context[
                            0, old_cur_len:old_cu_len
                        ]
                        kv_inp[0, old_end_idx : old_end_idx + n_latents] = latents[i]

                        cur_len = cu_len
                        old_cur_len = old_cu_len
                else:
                    kv_inp = torch.cat([context, latents], dim=-2)

        else:
            kv_inp = latents

        key_states = self.k_proj(kv_inp)
        value_states = self.v_proj(kv_inp)

        # query_states = query_states.view(bsz, q_len, self.num_heads, self.head_dim)
        query_states = query_states.view(
            *latents.shape[:2], self.num_heads, self.head_dim
        )

        key_states = key_states.view(
            *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)
        value_states = value_states.view(
            *kv_inp.shape[:2], self.num_key_value_heads, self.head_dim
        ).transpose(1, 2)

        # kv_seq_len = key_states.shape[-2]
        # if past_key_value is not None:
        #     kv_seq_len += past_key_value[0].shape[-2]

        # if past_key_value is not None:
        #     # Activate slicing cache only if the config has a value `sliding_windows` attribute
        #     if (
        #         hasattr(self.config, "sliding_window")
        #         and kv_seq_len > self.config.sliding_window
        #     ):
        #         slicing_tokens = kv_seq_len - self.config.sliding_window

        #         past_key = past_key_value[0]
        #         past_value = past_key_value[1]

        #         past_key = past_key[:, :, slicing_tokens:, :].contiguous()
        #         past_value = past_value[:, :, slicing_tokens:, :].contiguous()

        #         if past_key.shape[-2] != self.config.sliding_window - 1:
        #             raise ValueError(
        #                 "past key must have a shape of (`batch_size, num_heads, self.config.sliding_window-1,"
        #                 f" head_dim`), got {past_key.shape}"
        #             )

        #         past_key_value = (past_key, past_value)

        #         if attention_mask is not None:
        #             attention_mask = attention_mask[:, slicing_tokens:]
        #             attention_mask = torch.cat(
        #                 [attention_mask, torch.ones_like(attention_mask[:, -1:])], dim=-1
        #             )

        #     key_states = torch.cat([past_key_value[0], key_states], dim=2)
        #     value_states = torch.cat([past_key_value[1], value_states], dim=2)

        past_key_value = (key_states, value_states) if use_cache else None

        # repeat k/v heads if n_kv_heads < n_heads
        key_states = repeat_kv(key_states, self.num_key_value_groups)
        value_states = repeat_kv(value_states, self.num_key_value_groups)
        dropout_rate = 0.0 if not self.training else self.attention_dropout

        # In PEFT, usually we cast the layer norms in float32 for training stability reasons
        # therefore the input hidden states gets silently casted in float32. Hence, we need
        # cast them back in float16 just to be sure everything works as expected.
        input_dtype = query_states.dtype
        if input_dtype == torch.float32:
            if torch.is_autocast_enabled():
                target_dtype = torch.get_autocast_gpu_dtype()
            # Handle the case where the model is quantized
            elif hasattr(self.config, "_pre_quantization_dtype"):
                target_dtype = self.config._pre_quantization_dtype
            else:
                target_dtype = self.q_proj.weight.dtype

            logger.warning_once(
                f"The input hidden states seems to be silently casted in float32, this might be related to"
                f" the fact you have upcasted embedding or layer norm layers in float32. We will cast back the input in"
                f" {target_dtype}."
            )

            query_states = query_states.to(target_dtype)
            key_states = key_states.to(target_dtype)
            value_states = value_states.to(target_dtype)

        # Reashape to the expected shape for Flash Attention
        key_states = key_states.transpose(1, 2)
        value_states = value_states.transpose(1, 2)

        attn_output = _flash_attention_forward(
            query_states,
            key_states,
            value_states,
            attention_mask,
            q_len,
            dropout=dropout_rate,
            position_ids=position_ids,
            sliding_window=None,
            is_causal=self.is_causal,
            use_top_left_mask=self._flash_attn_uses_top_left_mask,
            **kwargs,
        )

        attn_output = attn_output.reshape(
            bsz, q_len, self.num_heads * self.head_dim
        ).contiguous()
        attn_output = self.o_proj(attn_output)

        if not output_attentions:
            attn_weights = None

        return attn_output, attn_weights, past_key_value


IDEFICS2_PERCEIVER_ATTENTION_CLASSES = {
    # "eager": Idefics2PerceiverAttention,
    "flash_attention_2": Idefics2PerceiverFlashAttention2,
}


class Idefics2PerceiverLayer(nn.Module):
    def __init__(self, config, layer_idx: int):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.n_latents = config.resampler_n_latents
        self.depth = config.resampler_depth
        self.rms_norm_eps = config.rms_norm_eps
        self.is_cross_attn = config.is_cross_attn
        self.concat_latents_context = config.concat_latents_context

        self.input_latents_norm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.input_context_norm = (
            Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
            if self.is_cross_attn
            else torch.nn.Identity()
        )
        self.self_attn = IDEFICS2_PERCEIVER_ATTENTION_CLASSES[
            config._attn_implementation
        ](config, layer_idx=layer_idx)
        self.post_attention_layernorm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.pre_ff_layernorm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)
        self.post_ff_layernorm = Idefics2RMSNorm(
            self.hidden_size, eps=self.rms_norm_eps
        )
        self.mlp = Idefics2MLP(
            hidden_size=config.hidden_size,
            intermediate_size=config.hidden_size * 4,
            output_size=config.hidden_size,
            hidden_act=config.hidden_act,
        )

    def forward(
        self,
        latents: torch.Tensor,
        context: torch.Tensor | None = None,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
        past_key_value: tuple[torch.Tensor] | None = None,
        output_attentions: bool | None = False,
        use_cache: bool | None = False,
        **kwargs,
    ) -> tuple[torch.FloatTensor, tuple[torch.FloatTensor, torch.FloatTensor] | None]:
        """
        Args:
            latents (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            context (`torch.FloatTensor`): input to the layer of shape `(batch, seq_len, embed_dim)`
            attention_mask (`torch.FloatTensor`, *optional*): attention mask of size
                `(batch, sequence_length)` where padding elements are indicated by 0.
            output_attentions (`bool`, *optional*):
                Whether or not to return the attentions tensors of all attention layers. See `attentions` under
                returned tensors for more detail.
            use_cache (`bool`, *optional*):
                If set to `True`, `past_key_values` key value states are returned and can be used to speed up decoding
                (see `past_key_values`).
            past_key_value (`Tuple(torch.FloatTensor)`, *optional*): cached past key and value projection states
        """
        residual = latents

        latents = self.input_latents_norm(latents)
        context = self.input_context_norm(context)

        latents, self_attn_weights, present_key_value = self.self_attn(
            latents=latents,
            is_cross_attn=self.is_cross_attn,
            concat_latents_context=self.concat_latents_context,
            context=context,
            attention_mask=attention_mask,
            position_ids=position_ids,
            **kwargs,
        )
        latents = self.post_attention_layernorm(latents)
        latents = residual + latents
        residual = latents

        # latents = self.post_attention_layernorm(latents)
        latents = self.pre_ff_layernorm(latents)
        latents = self.mlp(latents)
        latents = self.post_ff_layernorm(latents)
        latents = residual + latents

        outputs = (latents,)

        if output_attentions:
            outputs += (self_attn_weights,)

        if use_cache:
            outputs += (present_key_value,)

        return outputs


IDEFICS2_INPUTS_DOCSTRING = r"""
    Args:
        context (`torch.FloatTensor` of shape `(batch_size, sequence_length, hidden_dim)`):
            The hidden states of the image after vision encoder and modality projection.
        attention_mask (`torch.Tensor` of shape `(batch_size, sequence_length)`, *optional*):
            Mask to avoid performing attention on padding token indices. Mask values selected in `[0, 1]`:

            - 1 for tokens that are **not masked**,
            - 0 for tokens that are **masked**.

            [What are attention masks?](../glossary#attention-mask)
"""


@add_start_docstrings(
    "Idefics2 perceiver resampler model that performs `depth` blocks of cross-attention with a fixed ",
    "`n_latents` inputs to decrease embedding sequence length. The Resampler acts as a form of learned pooling and ",
    "is derived from [Perceiver: General Perception with Iterative Attention](https://arxiv.org/abs/2103.03206)",
    IDEFICS2_START_DOCSTRING,
)
class Idefics2PerceiverResampler(Idefics2PreTrainedModel):
    _supports_sdpa = False
    config_class = Idefics2PerceiverConfig

    def __init__(self, config) -> None:
        super().__init__(config)
        self.hidden_size = config.hidden_size
        self.hidden_act = config.hidden_act
        self.n_latents = config.resampler_n_latents
        self.n_x_attn_layers = 1
        self.depth = config.resampler_depth
        self.n_self_attn_layers = self.depth - self.n_x_attn_layers
        self.rms_norm_eps = config.rms_norm_eps
        self.concat_latents_context = config.concat_latents_context

        # Create Latents for Perceiver
        self.latents = nn.Parameter(torch.randn(self.n_latents, self.hidden_size))

        # Create Transformer Blocks
        if not self.concat_latents_context:
            x_attn_config = deepcopy(config)
            x_attn_config.is_cross_attn = True
            self.layers = [
                Idefics2PerceiverLayer(x_attn_config, idx)
                for idx in range(self.n_x_attn_layers)
            ]

            config.is_cross_attn = False
            self.layers += [
                Idefics2PerceiverLayer(config, idx)
                for idx in range(self.n_self_attn_layers)
            ]
            self.layers = nn.ModuleList(self.layers)
        else:
            self.layers = nn.ModuleList(
                [Idefics2PerceiverLayer(config, idx) for idx in range(self.depth)]
            )
        self.norm = Idefics2RMSNorm(self.hidden_size, eps=self.rms_norm_eps)

        self._use_flash_attention_2 = config._attn_implementation == "flash_attention_2"
        assert self._use_flash_attention_2

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ) -> torch.Tensor:
        # seq embed -> bsz seq embed
        if position_ids is None:
            bsz = context.shape[0]
        else:
            # flattened packed sequence
            bsz = torch.where(position_ids == 0, 1, 0).sum()

        latents = self.latents.unsqueeze(0).expand((bsz, *self.latents.size()))

        # latent_attention_mask = torch.ones(
        #     (attention_mask.size(0), latents.size(1)),
        #     dtype=attention_mask.dtype,
        #     device=attention_mask.device,
        # )
        # attention_mask = torch.cat([attention_mask, latent_attention_mask], dim=-1)
        attention_mask = (
            _prepare_4d_attention_mask(
                attention_mask, latents.dtype, tgt_len=self.n_latents
            )
            if not self._use_flash_attention_2
            else attention_mask
        )

        compressed_context = latents

        cu_seq_lens_q = torch.tensor(
            [self.n_latents] * (bsz + 1), device=context.device, dtype=torch.int32
        ) * torch.arange(bsz + 1, device=context.device, dtype=torch.int32)
        max_length_q = self.n_latents
        # cu_seq_lens_k = None
        # max_length_k = None
        if attention_mask is not None:
            logger.warning_once("Using attention mask for resampler")
            seq_lens_k = attention_mask.sum(dim=-1, dtype=torch.int32)
            cu_seq_lens_k = torch.cumsum(seq_lens_k, dim=0, dtype=torch.int32)
            max_length_k = seq_lens_k.max().item()
            cu_seq_lens_k = nn.functional.pad(cu_seq_lens_k, (1, 0))

        elif position_ids is not None:
            logger.warning_once("Using position ids for resampler")
            # [bsz + 1]
            # cu_seq_lens_q = torch.tensor(
            #     [self.n_latents] * (bsz + 1), device=context.device, dtype=torch.int32
            # ) * torch.arange(bsz + 1, device=context.device, dtype=torch.int32)
            # max_length_q = self.n_latents

            position_ids = position_ids.flatten()
            indices = torch.arange(
                position_ids.size(0), device=position_ids.device, dtype=torch.int32
            )
            # [bsz + 1]
            cu_seq_lens_k = torch.cat(
                (
                    indices[position_ids == 0],
                    torch.tensor(
                        position_ids.size(),
                        device=position_ids.device,
                        dtype=torch.int32,
                    ),
                )
            )

            max_length_k = position_ids.max() + 1
            if self.concat_latents_context:
                last_pos_ids = torch.arange(
                    position_ids[-1] + 1 + self.n_latents,
                    device=position_ids.device,
                    dtype=position_ids.dtype,
                )
                new_position_ids = torch.empty(
                    cu_seq_lens_k[-1] + self.n_latents * bsz,
                    dtype=position_ids.dtype,
                    device=position_ids.device,
                )

                cur_idx = 0
                for i, idx in enumerate(indices[position_ids == 0][1:]):
                    l = idx - cur_idx

                    new_cur_idx = cur_idx + self.n_latents * i
                    n_new_ids = l + self.n_latents
                    new_ids = torch.arange(
                        n_new_ids,
                        device=position_ids.device,
                        dtype=position_ids.dtype,
                    )
                    new_position_ids[new_cur_idx : new_cur_idx + n_new_ids] = new_ids
                    cur_idx = idx
                new_cur_idx = cur_idx + self.n_latents * (i + 1)
                new_position_ids[new_cur_idx:] = last_pos_ids
                position_ids = new_position_ids

        else:
            raise ValueError("either position_ids or attention_mask is required")

        if self.concat_latents_context:
            old_cu_seq_lens_k = cu_seq_lens_k
            cu_seq_lens_k = cu_seq_lens_k + cu_seq_lens_q
            max_length_k = max_length_q + max_length_k
            # print("Using concat latents + context for resampler")
            # print(f"updated position_ids: {position_ids}")
            # print(f"updated cu_seq_lens_k: {cu_seq_lens_k}")
            # print(f"updated max_length_k: {max_length_k}")
            # print(f"cu_seq_lens_q: {cu_seq_lens_q}")
            # print(f"max_length_q: {max_length_q}")
            for layer in self.layers:
                layer_outputs = layer(
                    compressed_context,
                    context,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cu_seq_lens_q=cu_seq_lens_q,
                    cu_seq_lens_k=cu_seq_lens_k,
                    max_length_q=max_length_q,
                    max_length_k=max_length_k,
                    old_cu_seq_lens_k=old_cu_seq_lens_k,
                )
                compressed_context = layer_outputs[0]
        else:
            for encoder in self.layers[: self.n_x_attn_layers]:
                # print("Using cross-attention for resampler")
                # print(f"position_ids: {position_ids}")
                # print(f"cu_seq_lens_q: {cu_seq_lens_q}")
                # print(f"max_length_q: {max_length_q}")
                # print(f"cu_seq_lens_k: {cu_seq_lens_k}")
                # print(f"max_length_k: {max_length_k}")
                layer_outputs = encoder(
                    compressed_context,
                    context,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cu_seq_lens_q=cu_seq_lens_q,
                    cu_seq_lens_k=cu_seq_lens_k,
                    max_length_q=max_length_q,
                    max_length_k=max_length_k,
                )

                compressed_context = layer_outputs[0]

            position_ids = torch.arange(
                self.n_latents, device=position_ids.device, dtype=torch.int32
            ).repeat(1, bsz)

            for processor in self.layers[self.n_x_attn_layers :]:
                cu_seq_lens_k = cu_seq_lens_q
                max_length_k = max_length_q
                # print(f"Using self-attention for resampler")
                # print(f"updated position_ids: {position_ids}")
                # print(f"updated cu_seq_lens_k: {cu_seq_lens_k}")
                # print(f"updated max_length_k: {max_length_k}")
                # print(f"cu_seq_lens_q: {cu_seq_lens_q}")
                # print(f"max_length_q: {max_length_q}")
                layer_outputs = processor(
                    compressed_context,
                    context=None,
                    attention_mask=None,
                    position_ids=position_ids,
                    past_key_value=None,
                    output_attentions=False,
                    use_cache=False,
                    cu_seq_lens_q=cu_seq_lens_q,
                    cu_seq_lens_k=cu_seq_lens_k,
                    max_length_q=max_length_q,
                    max_length_k=max_length_k,
                )

                compressed_context = layer_outputs[0]

        compressed_context = self.norm(compressed_context)

        return compressed_context


class Idefics2Perceiver(Idefics2PreTrainedModel):
    def __init__(
        self,
        encoder_config: Idefics2PerceiverConfig,
        decoder_config: Idefics2PerceiverConfig,
    ):
        super().__init__(encoder_config)
        self.modality_projection = Idefics2MLP(
            hidden_size=encoder_config.input_size,
            intermediate_size=encoder_config.intermediate_size_factor
            * encoder_config.input_size,
            output_size=encoder_config.hidden_size,
            hidden_act=encoder_config.hidden_act,
        )
        self.encoder = Idefics2PerceiverResampler._from_config(encoder_config)
        self.decoder = Idefics2PerceiverResampler._from_config(decoder_config)

    def forward(
        self,
        context: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.LongTensor | None = None,
    ):
        if position_ids is None:
            bsz = context.shape[0]
        else:
            bsz = torch.where(position_ids == 0, 1, 0).sum()
        projected_inputs = self.modality_projection(context)

        # [bsz, n_latents, dim]
        latents = self.encoder(
            context=projected_inputs,
            attention_mask=attention_mask,
            position_ids=position_ids,
        )

        # if position_ids is None:
        #     # padded
        #     # attn_mask = torch.ones(
        #     #     (bsz, self.encoder.n_latents),
        #     #     device=context.device,
        #     #     dtype=torch.int32,
        #     # )
        #     outputs = self.decoder(latents)
        # else:
        # packed sequence
        # always use position_ids for decoder
        latent_position_ids = torch.arange(
            self.encoder.n_latents, device=context.device
        ).unsqueeze(0)
        latent_position_ids = torch.tile(latent_position_ids, (1, bsz))
        outputs = self.decoder(latents, position_ids=latent_position_ids)

        return outputs


__all__ = [
    "Idefics2Perceiver",
]

if __name__ == "__main__":
    encoder_config = Idefics2PerceiverConfig(
        concat_latents_context=True,
        input_size=1234,
        resampler_n_latents=64,
        resampler_depth=1 + 8,
        attn_implementation="flash_attention_2",
    )
    decoder_config = Idefics2PerceiverConfig(
        concat_latents_context=True,
        input_size=encoder_config.hidden_size,
        resampler_n_latents=16,
        resampler_depth=1,
        attn_implementation="flash_attention_2",
    )
    model = (
        Idefics2Perceiver(encoder_config, decoder_config).to("cuda").to(torch.bfloat16)
    )
    # print(model)
    inp = torch.rand(1, 192 + 37, 1234, dtype=torch.bfloat16, device="cuda")
    inp = torch.cat([inp, inp], dim=1)
    # mask = torch.ones(1, 128, dtype=torch.bfloat16, device="cuda")
    # mask[..., -1] = 0
    # print(mask)
    out = model(
        inp,
        # attention_mask=mask,
        position_ids=torch.cat(
            [
                torch.arange(128, device="cuda"),
                torch.arange(64, device="cuda"),
                torch.arange(37, device="cuda"),
                torch.arange(128, device="cuda"),
                torch.arange(64, device="cuda"),
                torch.arange(37, device="cuda"),
            ],
            dim=-1,
        ).unsqueeze(0),
        # position_ids=torch.arange(128, device="cuda"),
    )
    print(out)
    print(out.shape)
    print(model.encoder.latents.grad)
    print(model.decoder.latents.grad)
    out.mean().backward()
    print(model.encoder.latents.grad)
    print(model.decoder.latents.grad)
    breakpoint()

    # inp = torch.rand(1, 192 + 37, 1234, dtype=torch.bfloat16, device="cuda")
    # inp = torch.cat([inp, inp], dim=1)
    # inp = inp.unsqueeze(1)
    # # mask = torch.ones(1, 128, dtype=torch.bfloat16, device="cuda")
    # # mask[..., -1] = 0
    # # print(mask)
    # out = torch.stack(
    #     [
    #         model(
    #             inp[:, i],
    #             # attention_mask=mask,
    #             position_ids=torch.cat(
    #                 [
    #                     torch.arange(128, device="cuda"),
    #                     torch.arange(64, device="cuda"),
    #                     torch.arange(37, device="cuda"),
    #                     torch.arange(128, device="cuda"),
    #                     torch.arange(64, device="cuda"),
    #                     torch.arange(37, device="cuda"),
    #                 ],
    #                 dim=-1,
    #             ).unsqueeze(0),
    #             # position_ids=torch.arange(128, device="cuda"),
    #         )
    #         for i in range(inp.shape[1])
    #     ]
    # )

    # print(out)
    # print(out.shape)
    # print(model.encoder.latents.grad)
    # print(model.decoder.latents.grad)
    # out.mean().backward()
    # print(model.encoder.latents.grad)
    # print(model.decoder.latents.grad)
    # breakpoint()
