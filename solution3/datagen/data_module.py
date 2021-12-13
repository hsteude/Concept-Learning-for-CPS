import pytorch_lightning as pl
from torch.utils.data import DataLoader, random_split
from solution3.datagen.dataset import ThreeTankDataSet
import numpy as np
import torch
from torch.utils.data import DataLoader


class ThreeTankDataModule(pl.LightningDataModule):
    def __init__(self, validdation_split, batch_size, dl_num_workers, *args, **kwargs):
        self.validdation_split = validdation_split
        self.batch_size = batch_size
        self.dl_num_workers = dl_num_workers
        super().__init__()

    def setup(self, stage=None):
        dataset_full = ThreeTankDataSet()
        dataset_size = len(dataset_full)
        len_val = int(np.floor(dataset_size * self.validdation_split))
        len_train = dataset_size - len_val
        self.dataset_train, self.dataset_val = random_split(
            dataset=dataset_full,
            lengths=[len_train, len_val],
            generator=torch.Generator(),
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            shuffle=True,
            batch_size=self.batch_size,
            num_workers=self.dl_num_workers,
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            shuffle=False,
            batch_size=self.batch_size,
            num_workers=self.dl_num_workers,
            pin_memory=True,
        )
