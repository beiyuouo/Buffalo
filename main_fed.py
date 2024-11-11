import os
import time
import torch
import ezkfg as ez
import numpy as np
from loguru import logger
from copy import deepcopy
from lightning.pytorch import seed_everything
from model.module.callbacks import add_callbacks
import lightning.pytorch as pl
from model.mrg import MRGModel
from dataset.data_module import DataModule
from utils import get_mean_iuxray, get_mean_mvqa

def train_client(cfg, client_idx=None, round_idx=0, params=None, client_type=None, **kwargs):
    cfg = deepcopy(cfg)
    # cfg.save_dir = os.path.join(cfg.save_dir, )
    os.makedirs(cfg.save_dir, exist_ok=True)
    cfg.client_idx = client_idx
    cfg.round_idx = round_idx
    cfg.client_type = client_type

    dm = DataModule(cfg, client_idx=client_idx)
    callbacks = add_callbacks(cfg)

    trainer = pl.Trainer(
        devices=cfg.n_gpus,
        num_nodes=cfg.n_nodes,
        strategy=cfg.strategy,
        accelerator=cfg.accelerator,
        precision=cfg.precision,
        val_check_interval=cfg.train.val_check_interval,
        limit_val_batches=cfg.train.limit_val_batches,
        max_epochs=cfg.train.num_epochs,
        num_sanity_val_steps=cfg.train.num_sanity_val_steps,
        accumulate_grad_batches=cfg.train.accumulate_grad_batches,
        check_val_every_n_epoch=cfg.train.check_val_every_n_epoch,
        callbacks=callbacks["callbacks"],
        logger=callbacks["loggers"],
    )

    # build model architecture
    model = MRGModel(cfg) # if cfg.task.type == "mrg" else MVQAModel(cfg)
    # logger.info(f"Model: {model}")

    if params is not None:
        model.set_trainable_params(params)

    if client_idx == None:
        logger.info(f"Start testing model at round {round_idx}")
        trainer.test(model, datamodule=dm)
    else:
        logger.info(f"Start training model at round {round_idx}, client {client_idx}")

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype" and round_idx != 0:
            # model.automatic_optimization = False
            if client_type == 0:
                logger.debug(f"fine-tune projector with prototype")
                # model.freeze_encoder()
                model.set_stage(1)
            else:
                logger.debug(f"fine-tune vision encoder with prototype")
                # model.unfreeze_encoder()
                if cfg.task.type == "vqa" and client_type == 1:
                    model.freeze_encoder()
                model.set_stage(2)

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "mean":
            if cfg.task.type == "vqa" and client_type == 1:
                model.freeze_encoder()

        trainer.fit(model, datamodule=dm)

        if model.freeze_ve:
            model.unfreeze_encoder()

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype":
            logger.info(f"Get prototype for client {client_idx}")
            return model.get_trainable_params(), model.get_prototype(dm)

    return model.get_trainable_params()


def main(cfg_file=os.path.join(os.path.dirname(__file__), "config", "mrg_config.yaml")):
    cfg = ez.load(cfg_file) if isinstance(cfg_file, str) else cfg_file
    cfg.save_dir = os.path.join(cfg.save_dir, f"{cfg.data.dataset}_{cfg.train.algor}", f"{time.strftime('%Y-%m-%d-%H-%M-%S')}")
    os.makedirs(cfg.save_dir, exist_ok=True)
    log_dir = cfg.save_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, f"train-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"))

    seed_everything(cfg.seed, workers=True)

    global_params = None
    global_prototype = None

    get_mean = get_mean_iuxray if cfg.task.type == "mrg" else get_mean_mvqa

    client_ids = list(range(cfg.train.num_clients))
    client_types = np.random.choice(list(range(cfg.train.num_types)), size=cfg.train.num_clients, replace=True)

    # make sure must have at least one client for each type
    for type_idx in range(cfg.train.num_types):
        if type_idx not in client_types:
            client_types[np.random.choice(client_ids)] = type_idx

    for type_idx in range(cfg.train.num_types):
        assert type_idx in client_types

    logger.info(f"Client types: {client_types}")

    for round_idx in range(cfg.train.num_rounds):
        logger.info(f"Training round {round_idx}")
        client_ids = list(range(cfg.train.num_clients))
        # select clients for this round
        client_ids = np.random.choice(client_ids, size=cfg.train.num_selected, replace=False)

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype" and round_idx == 0:
            client_ids = [i for i in range(cfg.train.num_clients) if client_types[i] == 0]

        # client_ids = [3, 4, 5, 6]
        logger.info(f"Selected clients: {client_ids}")

        client_params = []
        prototypes = []

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "mean":
            logger.info(f"Use strategy {cfg.train.mm_strategy}")
            means = get_mean(mean_type=cfg.train.mm_mean_type, cfg=cfg, global_params=global_params)
            # logger.debug(f"means[0].shape: {means[0].shape}, means[1].shape: {means[1].shape}, means[2].shape: {means[2].shape}")
            cfg["payload"] = {}
            cfg["payload"]["means"] = means

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype":
            logger.info(f"Use strategy {cfg.train.mm_strategy}")
            cfg["payload"] = {}
            cfg["payload"]["global_prototype"] = global_prototype
            if global_prototype is None:
                logger.info("No global prototype, use mean instead")
                means = get_mean(mean_type=cfg.train.mm_mean_type, cfg=cfg, global_params=global_params)
                cfg["payload"]["means"] = means

        for client_idx in client_ids:
            logger.info(f"Training client {client_idx}")

            if cfg.train.modal_hetero and cfg.train.mm_strategy == "drop" and client_types[client_idx] != 0:
                logger.info(f"Use strategy {cfg.train.mm_strategy} for client {client_idx}")
                continue

            if cfg.train.modal_hetero and cfg.train.mm_strategy == "duplicate" and client_types[client_idx] == 3:
                logger.info(f"Use strategy {cfg.train.mm_strategy} for client {client_idx}")
                continue

            client_param = train_client(cfg, client_idx=client_idx, round_idx=round_idx, params=global_params, client_type=client_types[client_idx])
            if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype":
                client_param, prototype = client_param
                prototypes.append(prototype)
            client_params.append(client_param)

        if cfg.train.modal_hetero and cfg.train.mm_strategy == "prototype":
            logger.info(f"Start aggregating prototypes")
            from algor.prototype import wfpa

            global_prototype = wfpa(cfg=cfg, prototypes=prototypes, round_idx=round_idx, client_ids=client_ids)

        # aggregate client parameters
        if cfg.train.algor == "fedavg":
            if len(client_params) == 0:
                logger.info("No client parameters to aggregate")
            else:
                from algor.fedavg import fedavg
                logger.info(f"Start aggregating client parameters using {cfg.train.algor}")
                global_params = fedavg(cfg=cfg, client_params=client_params, round_idx=round_idx, client_ids=client_ids, client_types=client_types)
        else:
            raise NotImplementedError(f"Algorithm {cfg.train.algor} is not implemented")

        # test model
        train_client(cfg, round_idx=round_idx, params=global_params) # comment this line to avoid testing


if __name__ == "__main__":
    # main(cfg_file=os.path.join(os.path.dirname(__file__), "config", "mrg_config.yaml"))
    main(cfg_file=os.path.join(os.path.dirname(__file__), "config", "mvqa_config.yaml"))
