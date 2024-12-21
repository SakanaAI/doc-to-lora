import logging
from contextlib import contextmanager
from typing import Iterable, Optional

import torch
from peft import PeftConfig, PeftModel
from peft.tuners.tuners_utils import BaseTunerLayer, check_target_module_exists
from peft.utils import get_peft_model_state_dict

logger = logging.getLogger(__name__)


# taken from https://discuss.pytorch.org/t/opinion-eval-should-be-a-context-manager/18998/3
@contextmanager
def evaluating(*models):
    """Temporarily switch to evaluation mode."""
    is_training = [model.training if model is not None else False for model in models]
    try:
        for model in models:
            if model is not None:
                model.eval()
        yield models
    finally:
        for model, training in zip(models, is_training):
            if model is not None:
                model.train(training)


def get_layers(model):
    if hasattr(model, "model"):
        return get_layers(model.model)
    return model.layers


def get_num_layers(model):
    return len(get_layers(model))


def get_num_params(model):
    total_params = 0
    trainable_params = 0
    for p in model.parameters():
        total_params += p.numel()
        if p.requires_grad:
            trainable_params += p.numel()

    return total_params, trainable_params


def log_num_train_params(model):
    logger.debug("Trainable model parameters:")
    for name, p in model.named_parameters():
        if p.requires_grad:
            logger.debug(f"{name}, dtype:{p.dtype}")

    num_total_params, num_trainable_params = get_num_params(model)
    logger.info(
        f"trainable params: {num_trainable_params:,d} "
        f"|| all params: {num_total_params:,d} "
        f"|| trainable%: {100 * num_trainable_params / num_total_params:.4f}"
    )


def get_peft_in_out_features(
    model: PeftModel,
    peft_config: Optional[PeftConfig] = None,
) -> tuple[dict[str, int], dict[str, int]]:

    if peft_config is None:
        peft_config = model.peft_config["default"]
    in_features = dict()
    out_features = dict()
    for module_name, module in model.named_modules():
        if not check_target_module_exists(peft_config, module_name):
            continue
        if not isinstance(module, BaseTunerLayer):
            continue
        # support just Linear layer for now
        # all modules should be a leave module that is Linear layer
        assert isinstance(
            module.base_layer, torch.nn.Linear
        ), "all modules should be a leave module that is Linear layer"

        # this should always pass
        name = module_name.split(".")[-1]
        assert name in peft_config.target_modules

        if name not in in_features:
            in_features[name] = module.in_features
            out_features[name] = module.out_features
        else:
            # assumes each module has the same input and output features
            assert in_features[name] == module.in_features
            assert out_features[name] == module.out_features

    return in_features, out_features


def generated_lora_to_state_dict(
    lora_dict: dict,
    module_names: dict,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict:
    lora_state_dict = dict()
    for target_module in target_modules:
        for layer_idx in layer_indices:
            for module_name in module_names[target_module][layer_idx]:
                if "lora_A" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["A"][layer_idx].cpu().contiguous()
                    )
                elif "lora_B" in module_name:
                    lora_state_dict[module_name] = (
                        lora_dict[target_module]["B"][layer_idx].cpu().contiguous()
                    )
                else:
                    raise ValueError(f"Unexpected module name: {module_name}")
    return lora_state_dict


def get_lora_module_names(
    model: PeftModel,
    target_modules: list[str],
    layer_indices: Iterable[int],
) -> dict[str, list[str]]:
    module_names = {
        target_module: [[] for _ in range(len(layer_indices))]
        for target_module in target_modules
    }
    for k in get_peft_model_state_dict(model):
        if "lora" not in k:
            continue
        layer_idx = int(k.split("layers.")[-1].split(".")[0])
        if layer_idx in layer_indices:
            for target_module in target_modules:
                if target_module in k:
                    module_names[target_module][layer_idx].append(k)
                    break
    return module_names
