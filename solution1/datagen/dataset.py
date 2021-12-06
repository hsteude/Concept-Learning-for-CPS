import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import solution1.constants as const


class ThreeTankDataSet(Dataset):
    def __init__(self):
        self.df = pd.read_parquet(const.DATA_PATH)
        x = torch.from_numpy(
            self.df[const.STATE_COL_NAMES].values.astype(np.float32))
        self.x = x / const.SCALING_CONST
        self.start_idx_list = [
            i * const.NUMBER_TIMESTEPS for i in range(const.NUMBER_OF_SAMPLES)
        ]
        self.end_idx_list = [
            i + const.NUMBER_TIMESTEPS for i in self.start_idx_list]
        self.labels = self.df[const.LABEL_COLS].values.astype(np.float32)

    def __len__(self):
        """Size of dataset"""
        return len(self.start_idx_list)

    def __getitem__(self, index):
        """Get one sample"""
        labels_out = self.labels[self.start_idx_list[index], :]
        return (
            self.x[self.start_idx_list[index]: self.end_idx_list[index], :],
            labels_out,
            index,
        )


if __name__ == "__main__":
    # test for lets have a look
    dataset = ThreeTankDataSet()
    idx = 10
    x, idx = dataset[idx]
    breakpoint()
