import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import torch
import solution2.constants as const


class ThreeTankDataSet(Dataset):
    """Write me!"""

    def __init__(self):
        self.df = pd.read_parquet(const.DATA_PATH)
        # scaler = StandardScaler()
        # self.df[const.STATE_COL_NAMES] = \
            # scaler.fit_transform(self.df[const.STATE_COL_NAMES])
        x = torch.from_numpy(self.df[const.STATE_COL_NAMES].values.astype(np.float32))
        self.x = x / const.SCALING_CONST
        self.start_idx_list = [i*const.NUMBER_TIMESTEPS for i in range(const.NUMBER_OF_SAMPLES)]
        self.end_idx_list = [i + const.NUMBER_TIMESTEPS for i in self.start_idx_list]
        self.labels = self.df[const.LABEL_COLS].values.astype(np.float32)
        self.answers  = np.log(self.df[const.ANSWER_COLS].values.astype(np.float32))



    def __len__(self):
        """Size of dataset
        """
        return len(self.start_idx_list)

    def __getitem__(self, index):
        """Get one sample"""
        x_out = self.x[self.start_idx_list[index]: self.end_idx_list[index], :]
        labels_out = self.labels[self.start_idx_list[index], :]
        answers_out = self.answers[self.start_idx_list[index], :]
        return x_out, answers_out, labels_out, index 

if __name__ == '__main__':
    breakpoint()
    dataset = ThreeTankDataSet()
    idx = 10
    x, answers, labels, idx = dataset[idx]
    breakpoint()
