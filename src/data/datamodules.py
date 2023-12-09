from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datasets import EnDeDataset


class EnDeDataModule(pl.LightningDataModule):
    def __init__(
        self,
        train_en_path,
        train_de_path,
        test_en_path,
        test_de_path,
        train_ratio=0.8,
        batch_size=8,
        num_workers=4,
        transform=None,
    ):
        super().__init__()
        self.train_en_path = train_en_path
        self.train_de_path = train_de_path
        self.test_en_path = test_en_path
        self.test_de_path = test_de_path

        self.train_ratio = train_ratio
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.transform = transform

    def prepare_data(self) -> None:
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            full_dataset = EnDeDataset(
                self.train_en_path, self.train_de_path, transform=self.transform
            )
            self.full_length = len(full_dataset)
            self.train_length = int(self.full_length * self.train_ratio)
            self.val_length = self.full_length - self.train_length
            self.train_dataset, self.val_dataset = random_split(
                full_dataset, [self.train_length, self.val_length]
            )

        if stage == "test" or stage is None:
            self.test_dataset = EnDeDataset(
                self.test_en_path, self.test_de_path, transform=self.transform
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
        )
