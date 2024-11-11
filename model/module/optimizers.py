import torch


def build_optimizer(cfg, model):
    ve_params = list(map(id, model.visual_encoder.parameters()))
    ed_params = filter(lambda x: id(x) not in ve_params, model.parameters())
    optimizer = getattr(torch.optim, cfg.train.optim)(
        [
            {"params": model.visual_encoder.parameters(), "lr": cfg.train.lr_ve},
            {"params": ed_params, "lr": cfg.train.lr_ed},
        ],
        # weight_decay=cfg.train.weight_decay,
        # amsgrad=cfg.train.amsgrad,
        # eps=1e-8, correct_bias=False
    )
    return optimizer
