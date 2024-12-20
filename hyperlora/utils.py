import logging

logger = logging.getLogger(__name__)


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
