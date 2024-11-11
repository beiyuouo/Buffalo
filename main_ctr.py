import os
import torch
import time
import ezkfg as ez
from loguru import logger
from lightning.pytorch import seed_everything
from model.module.callbacks import add_callbacks
import lightning.pytorch as pl
from model.mrg import MRGModel
from dataset.data_module import DataModule

def main(cfg_file=os.path.join(os.path.dirname(__file__), "config", "mrg_ctr_config.yaml")):
    cfg = ez.load(cfg_file)
    log_dir = cfg.save_dir
    os.makedirs(log_dir, exist_ok=True)
    logger.add(os.path.join(log_dir, f"train-{time.strftime('%Y-%m-%d-%H-%M-%S')}.log"))

    seed_everything(cfg.seed, workers=True)

    dm = DataModule(cfg, flag=True) # test during training
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
    model = MRGModel(cfg)
    logger.info(f"Model: {model}")

    trainer.fit(model, datamodule=dm)
    # trainer.test(model, datamodule=dm)


if __name__ == "__main__":
    main()
