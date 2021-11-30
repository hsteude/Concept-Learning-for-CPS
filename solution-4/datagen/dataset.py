from torch.utils.data import Dataset
import pandas as pd
import numpy as np


class ThreeTankStateDataset(Dataset):
    def __init__(self,
                 nb_of_samples: int = 10000,
                 window_size: int = 100,
                 start_idx: int = 0,
                 end_idx: int = 199,
                 ordered_samples=False):
        self.nb_of_samples = nb_of_samples
        self.window_size = window_size
        self.start_idx = start_idx
        self.end_idx = end_idx
        self.ordered_samples = ordered_samples

        self.df = pd.read_csv("../../data/solution_4_dataset.csv")

        self.samples = self._create_samples()

    def _create_samples(self):
        # create random number between 0 and len(df) - window size
        start_idx_max = self.end_idx - self.window_size
        start_idxs = np.random.randint(0, start_idx_max, self.nb_of_samples)
        if self.ordered_samples:
            start_idxs = np.sort(start_idxs)
        df0 = self.df.reset_index()
        return np.array(
            [df0.loc[i:i+self.window_size-1].values.astype(np.float32)
             for i in start_idxs])

    def __len__(self):
        """Size of dataset"""
        return self.samples.shape[0]

    def __getitem__(self, index):
        """Get one sample"""
        return self.samples[index, :, 1:]

