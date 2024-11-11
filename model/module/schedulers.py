import torch
from loguru import logger
import functools
from torch.optim.lr_scheduler import LambdaLR

def lr_lambda(current_step, num_warmup_steps, num_training_steps):
    if current_step < num_warmup_steps:
        return float(current_step) / float(max(1, num_warmup_steps))
    return max(0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps)))


def build_lr_scheduler(cfg, optimizer):
    if cfg.train.lr_scheduler.name == "StepLR":
        logger.debug(f"Using StepLR lr_scheduler, step_size={cfg.train.lr_scheduler.step_size}, gamma={cfg.train.lr_scheduler.gamma}")
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, cfg.train.lr_scheduler.step_size, cfg.train.lr_scheduler.gamma)
    elif cfg.train.lr_scheduler.name == "CosineAnnealingLR":
        logger.debug(f"Using CosineAnnealingLR lr_scheduler, T_max={cfg.train.num_epochs}, eta_min={cfg.train.lr_scheduler.eta_min}")
        lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.num_epochs, eta_min=cfg.train.lr_scheduler.eta_min)
    elif cfg.train.lr_scheduler.name == "LambdaLR":
        lr_scheduler = LambdaLR(optimizer, functools.partial(lr_lambda, num_warmup_steps=cfg.train.lr_scheduler.num_warmup_steps, num_training_steps=cfg.train.num_training_steps))
    else:
        raise NotImplementedError

    return lr_scheduler
