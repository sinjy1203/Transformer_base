from pytorch_lightning.utilities.types import TRAIN_DATALOADERS
from torch.utils.data import DataLoader
from torch.utils.data import random_split
from torchtext import transforms
import pytorch_lightning as pl
import pyrootutils

pyrootutils.setup_root(__file__, indicator=".project-root", pythonpath=True)
from src.data.datasets import Multi30kDataset
from extras.constants import *


class Multi30kDataModule(pl.LightningDataModule):
    def __init__(
        self, batch_size=32, num_workers=4, src_transform=None, tgt_transform=None
    ):
        super().__init__()
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.src_transform = src_transform
        self.tgt_transform = tgt_transform

        self.totensor = transforms.ToTensor(padding_value=SPECIAL_TOKENS_IDX["PAD"])

    def collate_fn(self, batch):
        batch_src, batch_tgt = list(zip(*batch))

        batch_src = self.totensor(list(batch_src))
        batch_tgt = self.totensor(list(batch_tgt))

        return batch_src, batch_tgt

    def prepare_data(self):
        pass

    def setup(self, stage=None):
        if stage == "fit" or stage is None:
            self.train_dataset = Multi30kDataset(
                split="train",
                src_transform=self.src_transform,
                tgt_transform=self.tgt_transform,
            )
            self.val_dataset = Multi30kDataset(
                split="valid",
                src_transform=self.src_transform,
                tgt_transform=self.tgt_transform,
            )

        if stage == "test" or stage is None:
            self.test_dataset = Multi30kDataset(
                split="test",
                src_transform=self.src_transform,
                tgt_transform=self.tgt_transform,
            )

    def train_dataloader(self):
        return DataLoader(
            self.train_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def val_dataloader(self):
        return DataLoader(
            self.val_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )

    def test_dataloader(self):
        return DataLoader(
            self.test_dataset,
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=self.num_workers,
            pin_memory=True,
            collate_fn=self.collate_fn,
        )


# class EnDeDataModule(pl.LightningDataModule):
#     def __init__(
#         self,
#         train_en_path,
#         train_de_path,
#         test_en_path,
#         test_de_path,
#         train_ratio=0.8,
#         batch_size=8,
#         num_workers=4,
#         transform=None,
#     ):
#         super().__init__()
#         self.train_en_path = train_en_path
#         self.train_de_path = train_de_path
#         self.test_en_path = test_en_path
#         self.test_de_path = test_de_path

#         self.train_ratio = train_ratio
#         self.batch_size = batch_size
#         self.num_workers = num_workers

#         self.transform = transform

#     def prepare_data(self) -> None:
#         pass

#     def setup(self, stage=None):
#         if stage == "fit" or stage is None:
#             full_dataset = EnDeDataset(
#                 self.train_en_path, self.train_de_path, transform=self.transform
#             )
#             self.full_length = len(full_dataset)
#             self.train_length = int(self.full_length * self.train_ratio)
#             self.val_length = self.full_length - self.train_length
#             self.train_dataset, self.val_dataset = random_split(
#                 full_dataset, [self.train_length, self.val_length]
#             )

#         if stage == "test" or stage is None:
#             self.test_dataset = EnDeDataset(
#                 self.test_en_path, self.test_de_path, transform=self.transform
#             )

#     def train_dataloader(self):
#         return DataLoader(
#             self.train_dataset,
#             batch_size=self.batch_size,
#             shuffle=True,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )

#     def val_dataloader(self):
#         return DataLoader(
#             self.val_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )

#     def test_dataloader(self):
#         return DataLoader(
#             self.test_dataset,
#             batch_size=self.batch_size,
#             shuffle=False,
#             num_workers=self.num_workers,
#             pin_memory=True,
#         )
