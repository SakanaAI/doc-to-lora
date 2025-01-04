import logging
from contextlib import contextmanager
from dataclasses import dataclass, field
from enum import Enum
from functools import partial, wraps
from math import log, pi
from typing import Any, Iterable, Optional, Tuple, Union

import torch
import torch.nn.functional as F

from configs import AggregatorArguments, HypernetArguments
from einops import rearrange, repeat, unpack
from einops.layers.torch import EinMix as Mix
from einops.layers.torch import Reduce
from hooks import add_generated_lora_hook, remove_hook_handles
from jaxtyping import Float, Integer
from model_loading import get_lora_config, get_model_and_tokenizer
from peft import get_peft_config, load_peft_weights, LoraConfig, PeftConfig, PeftModel
from peft.tuners._buffer_dict import BufferDict
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from pooling import POOL_FN, get_pooling_fn
from torch import Tensor, einsum, nn
from transformers import PerceiverConfig, PerceiverModel, PreTrainedModel
from transformers.models.perceiver.modeling_perceiver import (
    PerceiverBasicDecoder,
)
from transformers.modeling_outputs import ModelOutput
from utils import get_lora_module_names, get_num_layers, get_peft_in_out_features

logger = logging.getLogger()


class AGGREGATOR_TYPE(str, Enum):
    POOLER = "pooler"
    PERCEIVER = "perceiver"


@dataclass
class AggregatorConfig:
    aggregator_type: AGGREGATOR_TYPE
    
    # pooler
    pooling_type: POOL_FN
    feature_size: int
    num_layers: int
    num_modules: int
    output_size: int

    # perceiver
    attention_probs_dropout_prob: float = 0.0
    num_blocks: int = 1
    num_self_attends_per_block: int = 16
    self_attention_widening_factor: int = 4
    cross_attention_widening_factor: int = 1


def get_aggregator_config(
    model: PreTrainedModel, output_size: int, aggregator_args: AggregatorArguments
):
    lora_config = model.peft_config["default"]
    return AggregatorConfig(
        feature_size=model.config.hidden_size,
        output_size=output_size,
        num_layers=get_num_layers(model),
        num_modules=len(lora_config.target_modules),
        **vars(aggregator_args),
    )


@dataclass
class HypernetConfig:
    latent_size: int
    lora_config: LoraConfig
    module_names: dict[str, list[str]]
    layer_indices: Iterable[int]
    feature_sizes: tuple[dict[str, int], dict[str, int]]
    aggregator_config: AggregatorConfig


def get_hypernet_config(
    model: PreTrainedModel,
    hypernet_args: HypernetArguments,
    aggregator_args: AggregatorArguments,
):
    lora_config = model.peft_config["default"]
    indices = torch.arange(get_num_layers(model), device=model.device)
    return HypernetConfig(
        latent_size=hypernet_args.latent_size,
        lora_config=lora_config,
        module_names=get_lora_module_names(model, lora_config.target_modules, indices),
        layer_indices=indices,
        feature_sizes=get_peft_in_out_features(model, peft_config=lora_config),
        aggregator_config=get_aggregator_config(
            model,
            hypernet_args.latent_size,
            aggregator_args,
        ),
    )


class Perceiver(nn.Module):
    """perceiver w/ bottleneck size = n_modules * n_layers"""

    def __init__(
        self, feature_size, output_size, num_layers, num_modules, *args, **kwargs
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules
        self.config = PerceiverConfig(
            d_model=feature_size,  # + num_bands
            num_latents=num_layers * num_modules * 8,
            d_latents=output_size,
            # attention_probs_dropout_prob=0.0,
            # num_blocks=8,
            # num_self_attends_per_block=6,
            # self_attention_widening_factor=4,
            **kwargs,
        )
        decoder = PerceiverBasicDecoder(
            self.config,
            output_num_channels=output_size,
            output_index_dims=num_layers * num_modules,
            num_channels=output_size,
            final_project=False,
            trainable_position_encoding_kwargs=dict(
                num_channels=output_size,
                index_dims=num_layers * num_modules,
            ),
        )

        self.perceiver = PerceiverModel(self.config, decoder=decoder)

    def forward(
        self,
        ctx_features: Float[Tensor, "bs seq_len feature_dim"],
        ctx_attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        x = self.perceiver(ctx_features, ctx_attn_mask).logits  # .last_hidden_state
        x = rearrange(
            x,
            "bs (n_layers n_modules) d -> bs n_layers n_modules d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
        )
        return x


class Mixer(nn.Module):
    def __init__(
        self,
        input_size: int,
        intermediate_emb_size: int,
        output_size: int,
    ):
        super().__init__()
        self.gate_proj = nn.Linear(input_size, intermediate_emb_size, bias=False)
        self.up_proj = nn.Linear(input_size, intermediate_emb_size, bias=False)
        self.down_proj = nn.Linear(intermediate_emb_size, output_size, bias=False)
        self.act_fn = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.down_proj(self.act_fn(self.gate_proj(x)) * self.up_proj(x))


class MLPResidualBlock(nn.Module):
    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        output_size: int,
        pre_layer_norm: bool = True,
        post_dropout: bool = True,
    ):
        super().__init__()
        layers = []
        if pre_layer_norm:
            layers.append(nn.LayerNorm(input_size))
        layers += [
            nn.Dropout(0.05),
            nn.Linear(input_size, hidden_size),
            nn.SiLU(),
            nn.Dropout(0.05),
            nn.Linear(hidden_size, output_size),
            nn.SiLU(),
        ]
        if post_dropout:
            layers.append(nn.Dropout(0.05))
        self.mlp = nn.Sequential(*layers)

    def forward(self, x):
        return x + self.mlp(x)


class Pooler(nn.Module):
    def __init__(
        self,
        feature_size: int,
        output_size: int,
        pooling_type: POOL_FN,
        num_layers: int,
        num_modules: int,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.num_layers = num_layers
        self.num_modules = num_modules

        # NOTE: features will be projected to size = output_size // 2
        # then cat with layer and module embeddings (each with size output_size // 4)
        # which are collectively form features with size = output_size
        self.pool_fn = get_pooling_fn(pooling_type)
        self.feature_proj = nn.Linear(feature_size, output_size // 2)
        self.ln = nn.LayerNorm(output_size // 2)
        self.layer_embs = nn.Sequential(
            nn.Embedding(num_layers, output_size // 4),
            nn.LayerNorm(output_size // 4),
        )
        self.module_embs = nn.Sequential(
            nn.Embedding(num_modules, output_size // 4),
            nn.LayerNorm(output_size // 4),
        )
        self.mixer = Mixer(output_size, output_size * 4, output_size)
        self.mlp = MLPResidualBlock(output_size, output_size * 4, output_size)

        self.register_buffer("layer_indices", torch.arange(num_layers))
        self.register_buffer("module_indices", torch.arange(num_modules))

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        bs = features.shape[0]

        # [bs, feature_dim]
        x = self.ln(self.feature_proj(self.pool_fn(features, attn_mask).float()))
        x = repeat(
            x,
            "bs d -> bs n_layers n_modules d",
            n_modules=self.num_modules,
            n_layers=self.num_layers,
        )

        layer_embs = self.layer_embs(self.layer_indices)  # [num_layers, d]
        layer_embs = repeat(
            layer_embs,
            "n_layers d -> bs n_layers n_modules d",
            bs=bs,
            n_modules=self.num_modules,
        )

        module_embs = self.module_embs(self.module_indices)  # [num_modules, d]
        module_embs = repeat(
            module_embs,
            "n_modules d -> bs n_layers n_modules d",
            bs=bs,
            n_layers=self.num_layers,
        )

        emb = torch.cat([x, layer_embs, module_embs], dim=3)
        return self.mlp(self.mixer(emb))


AG = {
    AGGREGATOR_TYPE.POOLER: Pooler,
    AGGREGATOR_TYPE.PERCEIVER: Perceiver,
}


@contextmanager
def early_exit(base_model: PreTrainedModel, exit_layer: int):
    try:
        layers = base_model.layers
        base_model.layers = layers[:exit_layer]
        yield base_model
    finally:
        base_model.layers = layers


@contextmanager
def maybe_add_batch_dim(kwargs):
    try:
        batched_input = False
        batched_attn_mask = False
        if len(kwargs["input_ids"].shape) == 1:
            kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
            batched_input = True
        if len(kwargs["attention_mask"].shape) == 1:
            kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)
            batched_attn_mask = True
        yield batched_input, batched_attn_mask
    finally:
        if batched_input:
            kwargs["input_ids"] = kwargs["input_ids"].squeeze(0)
        if batched_attn_mask:
            kwargs["attention_mask"] = kwargs["attention_mask"].squeeze(0)


class EarlyExit(nn.Module):
    def __init__(self, base_model: PreTrainedModel, exit_layer: int):
        super().__init__()
        self.base_model = base_model
        self.exit_layer = exit_layer

    def forward(self, **kwargs):
        # if len(kwargs["input_ids"].shape) == 1:
        #     kwargs["input_ids"] = kwargs["input_ids"].unsqueeze(0)
        # if len(kwargs["attention_mask"].shape) == 1:
        #     kwargs["attention_mask"] = kwargs["attention_mask"].unsqueeze(0)

        with (
            early_exit(self.base_model, self.exit_layer),
            maybe_add_batch_dim(kwargs) as (batched_input, batched_attn_mask),
        ):
            model_outputs = self.base_model(**kwargs)

        if batched_input:
            model_outputs.last_hidden_state = model_outputs.last_hidden_state.squeeze(0)

        return model_outputs.last_hidden_state


def get_init_peft_weights(model: PeftModel, peft_config: PeftConfig = None):
    if peft_config is None:
        peft_config = model.peft_config["default"]
    peft_weights = {module_name: dict() for module_name in peft_config.target_modules}
    adapter_name = "default"
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(
            module.base_layer, nn.Linear
        ), "all modules should be a leave module that is Linear layer"

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        for submodule_name, submodule in module.named_modules():
            if not isinstance(submodule, (nn.ModuleDict, nn.ParameterDict, BufferDict)):
                continue

            if adapter_name not in submodule:
                continue

            if submodule_name not in peft_weights[name]:
                peft_weights[name][submodule_name] = submodule[adapter_name]
            else:
                smod1 = peft_weights[name][submodule_name]
                smod2 = submodule[adapter_name]
                assert type(smod1) == type(smod2)

    return peft_weights


class HyperLoRA(nn.Module):
    def __init__(
        self,
        # latent_size: int,
        # lora_config: LoraConfig,
        # layer_indices: Integer[Tensor, "num_layers"],
        # in_out_features: dict,
        # aggregator_type: AGGREGATOR_TYPE,
        # aggregator_kwargs: dict,
        config: HypernetConfig,
        base_model: Optional[PreTrainedModel] = None,
    ):
        super().__init__()

        # aggregator output [bs, n_layers, n_modules, feature_dim]
        # by mixing the pooled features with layer embs and module embs (for pooling)
        # or via a perceiver w/ bottleneck size = n_modules * n_layers
        agg_config = config.aggregator_config
        self.aggregator = AG[agg_config.aggregator_type](**vars(agg_config))

        self.lora_config = config.lora_config

        self.target_modules = self.lora_config.target_modules
        self.layer_indices = config.layer_indices

        self.in_d, self.out_d = config.feature_sizes

        self.layers = MLPResidualBlock(
            input_size=config.latent_size,
            hidden_size=config.latent_size * 4,
            output_size=config.latent_size,
        )

        # TODO: check initialization of the head
        # default values are prob. way too big
        #
        # each module processes d -> r out_d
        # TODO: could be even more efficient if we use lightweight LoRA
        # ie. project the input to a smaller subspace (w/ same size) for all modules
        # have to also change in_d and out_d accordingly
        self.head = Mix(
            "bs n_layers n_modules d -> bs n_layers n_modules r out_d",
            weight_shape="n_modules d r out_d",
            # bias_shape=None,  # no bias
            bias_shape="n_modules r out_d",
            n_modules=len(self.target_modules),
            d=config.latent_size,
            r=config.lora_config.r,
            out_d=max(self.in_d[m] + self.out_d[m] for m in self.target_modules),
        )
        if base_model is not None:
            self._init_head(base_model)

    @torch.no_grad()
    def _init_head(self, base_model: PreTrainedModel):
        peft_weights = get_init_peft_weights(base_model, self.lora_config)
        logger.debug(f"peft_weights: {peft_weights}")
        self.head.weight.data[:] = 0
        self.head.bias.data[:] = 0
        for i, m in enumerate(self.target_modules):
            A = peft_weights[m]["lora_A"].weight.clone()  # [r, in_d]
            B = peft_weights[m]["lora_B"].weight.clone()  # [out_d, r]
            biases = [A, B.T]

            # bias_hyper_init(self.head.bias[i], biases)
            # bias-hyperinit
            # init weights to zeros and bias to the base weights
            bias_cat = torch.cat(biases, dim=1)
            self.head.bias.data[..., i, :, : bias_cat.shape[1]] = bias_cat
        self.head.bias.requires_grad = False

    def _to_lora_dict(
        self, flat_loras: Float[Tensor, "bs n_layers n_modules r max_io_dim"]
    ) -> dict[str, dict[str, Float[Tensor, "bs n_layers r _"]]]:
        # # list of [bs, n_layers, r, in_out_dim]
        # # and in_out_dim might vary across modules
        # loras = unpack(
        #     flat_loras,
        #     [[self.in_d[m] + self.out_d[m]] for m in self.target_modules],
        #     "bs n_layers r *",
        # )
        loras = unpack(
            flat_loras,
            [[] for _ in range(len(self.target_modules))],
            "bs n_layers * r max_io_dim",
        )

        # dict of {module:
        #   {A: [bs, n_layers, r, in_dim],
        #    B: [bs, n_layers, r, out_dim]}}
        lora_dict = dict()
        for module, lora in zip(self.target_modules, loras):
            A, B = unpack(
                lora[..., : self.in_d[module] + self.out_d[module]],
                [[self.in_d[module]], [self.out_d[module]]],
                "bs n_layers r *",
            )
            # transpose B
            B = rearrange(B, "bs n_layers r d_out -> bs n_layers d_out r")
            lora_dict[module] = dict(A=A, B=B)

        return lora_dict

    def forward(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):

        # [bs, n_layers, n_modules, feature_dim]
        emb = self.aggregator(features.to(torch.float32), attn_mask)

        # [bs, n_layers, n_modules, r, max_in_out_dim]
        flat_loras = self.head(self.layers(emb))

        return flat_loras

    def generate_loras(
        self,
        features: Float[Tensor, "bs seq_len feature_dim"],
        attn_mask: Optional[Integer[Tensor, "bs seq_len"]] = None,
    ):
        flat_loras = self.forward(features, attn_mask)
        return self._to_lora_dict(flat_loras)


class ModulatedPretrainedModel(nn.Module):
    def __init__(
        self,
        base_model: PreTrainedModel,  # TODO: maybe just use the name/config?
        hypernet: HyperLoRA,
        ctx_encoder: nn.Module,  # TODO: maybe just use the name/config?
    ):
        super().__init__()
        self.device = base_model.device
        # register base_model as a submodule
        self.register_module("base_model", base_model)
        self.register_module("hypernet", hypernet)
        self.register_module("ctx_encoder", ctx_encoder)

    # Delegate to base_model
    @property
    def config(self):
        return self.base_model.config

    def get_input_embeddings(self):
        return self.base_model.get_input_embeddings()

    def state_dict(self, *args, **kwargs):
        state_dict = super().state_dict(*args, **kwargs)
        # remove non-trainable and base_model's params
        for name, param in self.named_parameters():
            if not param.requires_grad:
                state_dict.pop(name, None)

        for name in list(state_dict.keys()):
            if name.startswith("base_model") or name.startswith("ctx_encoder"):
                state_dict.pop(name, None)

        return state_dict

    def load_state_dict(self, state_dict: dict, *args, **kwargs):
        # NOTE: might have to set `strict=False` as we don't save all the params
        return super().load_state_dict(state_dict, *args, **kwargs)

    # @torch.no_grad()
    # def get_ctx_features(
    #     self,
    #     examples: dict[str, Any],
    # ):
    #     # TODO: truncate the ctx_ids to the max_ctx_len
    #     out = dict()
    #     input_ids = torch.tensor(examples.get("ctx_ids")).to(self.device)
    #     # we don't really use ctx_attn_mask here but it'll be padded
    #     # and used by hypernet.aggregator which has to handle ctx_attn_mask
    #     attention_mask = torch.tensor(examples.get("ctx_attn_mask")).to(self.device)

    #     # NOTE: this might not work for batched inputs
    #     if isinstance(self.ctx_encoder, nn.Embedding):
    #         features = self.ctx_encoder(input_ids)
    #     else:
    #         features = self.ctx_encoder(
    #             input_ids=input_ids, attention_mask=attention_mask
    #         )
    #     out["ctx_features"] = features
    #     return out

    def forward(
        self,
        # ctx_features: Optional[Float[Tensor, "bs ctx_length feature_dim"]] = None,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        **model_inputs_kwargs: dict[str, Any],
    ) -> Union[tuple, ModelOutput]:
        """Forward pass of the modulated model."""

        generated_loras = None
        if ctx_ids is None:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_features provided, using the base model for forward pass\n\n",
                    "*" * 100,
                )
            )
            # model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            # return model_outputs

        else:
            with torch.no_grad():
                ctx_features = self.ctx_encoder(
                    input_ids=ctx_ids, attention_mask=ctx_attn_mask
                )
            generated_loras = self.hypernet.generate_loras(ctx_features, ctx_attn_mask)

        # apply lora hook to the base model
        # self.apply_generated_loras(generated_loras)
        with apply_generated_loras(
            self.base_model,
            generated_loras,
            self.hypernet.layer_indices,
            self.training,
        ):
            model_outputs = self.base_model(**model_inputs_kwargs)
        # model_outputs.generated_loras = generated_loras

        return model_outputs

    @torch.no_grad()
    def generate(
        self,
        ctx_ids: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        ctx_attn_mask: Optional[Integer[Tensor, "bs ctx_length"]] = None,
        **model_inputs_kwargs: dict[str, Any],
    ):
        generated_loras = None
        if ctx_ids is None:
            logger.warning(
                (
                    "*" * 100,
                    "\n\nNo ctx_ids provided, using the base model for generation\n\n",
                    "*" * 100,
                )
            )
            # model_outputs = self.base_model(**model_inputs_kwargs)
            # model_outputs.generated_loras = None
            # return model_outputs
        else:
            ctx_features = self.ctx_encoder(
                input_ids=ctx_ids, attention_mask=ctx_attn_mask
            )
            generated_loras = self.hypernet.generate_loras(ctx_features, ctx_attn_mask)

        # apply lora hook to the base model
        # self.apply_generated_loras(generated_loras)
        with apply_generated_loras(
            self.base_model,
            generated_loras,
            self.hypernet.layer_indices,
            self.training,
        ):
            model_outputs = self.base_model.generate(**model_inputs_kwargs)
        return model_outputs


@contextmanager
def apply_generated_loras(
    base_model: nn.Module,
    generated_loras: Optional[dict] = None,
    layer_indices: Optional[Iterable[int]] = None,
    training: bool = False,
):
    if generated_loras is None:
        yield base_model
        return

    try:
        hooks = []
        for module_name in generated_loras:
            for layer_idx in layer_indices:
                hooks += add_generated_lora_hook(
                    base_model,
                    module_name,
                    layer_idx,
                    A=generated_loras[module_name]["A"][:, layer_idx],
                    B=generated_loras[module_name]["B"][:, layer_idx],
                    scaling=base_model.peft_config["default"].lora_alpha,
                    input_dropout=base_model.peft_config["default"].lora_dropout,
                    training=training,
                )

        yield base_model
    finally:
        remove_hook_handles(hooks)


if __name__ == "__main__":
    from utils import get_base_model

    # set torch randomness seed
    torch.manual_seed(42)
    model_name = "meta-llama/Llama-3.2-1B-Instruct"
    base_model, tokenizer = get_model_and_tokenizer(
        model_name,
        train=True,
        requires_grad=False,
        peft_config=get_lora_config(model_name),
    )
    print(base_model)
    device = base_model.device
    # lora_config = base_model.peft_config["default"]
    # in_d, out_d = get_peft_in_out_features(base_model, peft_config=lora_config)

    # hypernet = HyperLoRA(
    #     lora_config,
    #     torch.arange(get_num_layers(base_model), device=base_model.device),
    #     in_out_features={"in": in_d, "out": out_d},
    #     aggregator_type=AGGREGATOR_TYPE.POOLER,
    #     aggregator_config=get_aggregator_config(base_model, POOL_FN.MEAN),
    # ).to(base_model.device)

    ctx_encoder = EarlyExit(get_base_model(base_model), 4)
    hypernet = HyperLoRA(get_hypernet_config(base_model), base_model).to(device)

    model = ModulatedPretrainedModel(base_model, hypernet, ctx_encoder).to(device)
    print(model)

    ctx_msg = "Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do eiusmod tempor incididunt ut labore et dolore magna aliqua. Ut enim ad minim veniam, quis nostrud exercitation ullamco laboris nisi ut aliquip ex ea commodo consequat. Duis aute irure dolor in reprehenderit in voluptate velit esse cillum dolore eu fugiat nulla pariatur. Excepteur sint occaecat cupidatat non proident, sunt in culpa qui officia deserunt mollit anim id est laborum."
    ctx_inputs = tokenizer(ctx_msg, return_tensors="pt").to(device)
    ctx_ids = ctx_inputs["input_ids"]
    ctx_attn_mask = ctx_inputs["attention_mask"]
    ctx_features = model.ctx_encoder(input_ids=ctx_ids, attention_mask=ctx_attn_mask)
    print(ctx_ids.shape)

    model.eval()
    agg_features = hypernet.aggregator(ctx_features, ctx_attn_mask)
    print(agg_features)
    print(agg_features.shape)

    hnetout = hypernet(ctx_features, ctx_attn_mask)
    print(hnetout)
    print(hnetout.shape)

    prompt_msg = "hello"
    prompt_inputs = tokenizer(prompt_msg, return_tensors="pt").to(device)

    basemodelout = model.base_model(**prompt_inputs)
    print(basemodelout)

    modelout = model(ctx_ids, ctx_attn_mask, **prompt_inputs)
    print(modelout)

    breakpoint()
