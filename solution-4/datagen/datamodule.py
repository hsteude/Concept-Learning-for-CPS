import pytorch_lightning as pl

from torch.utils.data import DataLoader
from dataset import ThreeTankStateDataset


class ThreeTankStateDataModule(pl.LightningDataModule):
    def __init__(self, nb_of_samples=10000, window_size=100, ordered_samples=False,
                 batch_size=32, num_workers=8, pin_memory=True,
                 train_split=0.7, val_split=0.1):
        super().__init__()
        self.nb_of_samples = nb_of_samples
        self.window_size = window_size
        self.ordered_samples = ordered_samples
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory

        self.train_split = train_split
        self.val_split = val_split
        self.train_dataset = None
        self.val_dataset = None
        self.test_dataset = None

    def prepare_data(self) -> None:
        len_data = 145000
        # [train || val || test]
        train_nb_samples = int(self.nb_of_samples * self.train_split)
        val_nb_samples = int(self.nb_of_samples * self.val_split)
        test_nb_samples = self.nb_of_samples - train_nb_samples - val_nb_samples

        val_start_idx = int(len_data * self.train_split)
        test_start_idx = int(len_data * (self.train_split + self.val_split))
        # purge samples to stop overlapping
        train_end_idx = val_start_idx - 2 * self.window_size
        val_end_idx = test_start_idx - 2 * self.window_size

        self.train_dataset = ThreeTankStateDataset(nb_of_samples=train_nb_samples,
                                                   window_size=self.window_size,
                                                   ordered_samples=self.ordered_samples,
                                                   start_idx=0,
                                                   end_idx=train_end_idx)
        self.val_dataset = ThreeTankStateDataset(nb_of_samples=val_nb_samples,
                                                 window_size=self.window_size,
                                                 ordered_samples=self.ordered_samples,
                                                 start_idx=val_start_idx,
                                                 end_idx=val_end_idx)
        self.test_dataset = ThreeTankStateDataset(nb_of_samples=test_nb_samples,
                                                  window_size=self.window_size,
                                                  ordered_samples=self.ordered_samples,
                                                  start_idx=test_start_idx,
                                                  end_idx=len_data - self.window_size)

    def train_dataloader(self) -> DataLoader:
        # [batch_size, seq_len, features]
        return DataLoader(self.train_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_dataset,
                          batch_size=self.batch_size,
                          num_workers=self.num_workers,
                          pin_memory=self.pin_memory)
