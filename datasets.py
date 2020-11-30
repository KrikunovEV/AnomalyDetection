import pickle
import numpy as np
from torch.utils.data.dataset import Dataset


class ECG5000Dataset(Dataset):

    def __init__(self, filename: str):
        with open(filename, 'rb') as f:
            self.data = pickle.load(f).astype(np.float32)

    def __len__(self):
        return self.data.shape[0]

    def __getitem__(self, item):
        return self.data[item]
