import os
import time
from loguru import logger
from lightning.pytorch.loggers import CSVLogger
from lightning.pytorch import loggers as pl_loggers
from lightning.pytorch.callbacks import LearningRateMonitor
from lightning.pytorch.callbacks import ModelCheckpoint


def add_callbacks(cfg):
    log_dir = cfg.save_dir
    os.makedirs(log_dir, exist_ok=True)

    # --------- Add Callbacks
    if getattr(cfg, "client_idx", None) is not None:
        client_idx = cfg.client_idx
        round_idx = cfg.round_idx
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename=f"client{client_idx}_round{round_idx}" + "_{epoch:02d}",
            save_top_k=-1,
            every_n_train_steps=cfg.train.every_n_train_steps,
            every_n_epochs=cfg.train.every_n_epochs,
            save_last=False,
            save_weights_only=False,
        )
    else:
        checkpoint_callback = ModelCheckpoint(
            dirpath=os.path.join(log_dir, "checkpoints"),
            filename="ctr" + "_{epoch:02d}",
            save_top_k=-1,
            every_n_train_steps=cfg.train.every_n_train_steps,
            every_n_epochs=cfg.train.every_n_epochs,
            save_last=False,
            save_weights_only=False,
        )
    
    logger.info(f"Checkpoint callback: {checkpoint_callback}")

    lr_monitor_callback = LearningRateMonitor(logging_interval="step")
    tb_logger = pl_loggers.TensorBoardLogger(save_dir=os.path.join(log_dir, "logs"), name="tensorboard")
    csv_logger = CSVLogger(save_dir=os.path.join(log_dir, "logs"), name="csvlog")

    to_returns = {"callbacks": [checkpoint_callback, lr_monitor_callback], "loggers": [csv_logger, tb_logger]}
    return to_returns
