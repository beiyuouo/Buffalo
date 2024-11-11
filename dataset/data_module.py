from lightning.pytorch import LightningDataModule
from torch.utils.data import DataLoader
from .data_helper import create_datasets


class DataModule(LightningDataModule):

    def __init__(self, cfg, client_idx=None, flag=False):
        super().__init__()
        self.cfg = cfg
        self.client_idx = client_idx
        self.flag = flag

    def prepare_data(self):
        """
        Use this method to do things that might write to disk or that need to be done only from a single process in distributed settings.

        download

        tokenize

        etc…
        :return:
        """

    def setup(self, stage: str):
        """
        There are also data operations you might want to perform on every GPU. Use setup to do things like:

        count number of classes

        build vocabulary

        perform train/val/test splits

        apply transforms (defined explicitly in your datamodule or assigned in init)

        etc…
        :param stage:
        :return:
        """
        train_dataset, val_dataset, test_dataset = create_datasets(self.cfg, client_idx=self.client_idx)
        if not self.flag:
            self.dataset = {
                "train": train_dataset,
                "validation": val_dataset,
                "test": test_dataset
            }
        else:
            self.dataset = {"train": train_dataset, "validation": test_dataset, "test": val_dataset}

    def train_dataloader(self):
        """
        Use this method to generate the train dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(
            self.dataset["train"],
            batch_size=self.cfg.train.batch_size,
            drop_last=True,
            pin_memory=True,
            num_workers=self.cfg.data.num_workers,
            prefetch_factor=self.cfg.data.prefetch_factor,
        )
        return loader

    def val_dataloader(self):
        """
        Use this method to generate the val dataloader. Usually you just wrap the dataset you defined in setup.
        :return:
        """
        loader = DataLoader(
            self.dataset["validation"],
            batch_size=self.cfg.train.val_batch_size,
            drop_last=False,
            pin_memory=True,
            num_workers=self.cfg.data.num_workers,
            prefetch_factor=self.cfg.data.prefetch_factor,
        )
        return loader

    def test_dataloader(self):
        loader = DataLoader(
            self.dataset["test"],
            batch_size=self.cfg.train.test_batch_size,
            drop_last=False,
            pin_memory=False,
            num_workers=self.cfg.data.num_workers,
            prefetch_factor=self.cfg.data.prefetch_factor,
        )
        return loader
