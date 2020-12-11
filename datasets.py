import pickle
import numpy as np
import os
from PIL import Image
from enum import Enum
from torch.utils.data.dataset import Dataset
from torchvision import transforms


class DatasetType(Enum):
    Train = 1
    Test = 2


class MVTecDataset(Dataset):

    def __init__(self, type: DatasetType, cfg):
        self.type = type
        self.transform = transforms.Compose([
            transforms.Resize(64),
            transforms.ToTensor()
        ])

        if type == DatasetType.Train:
            dir = os.path.join(cfg.train_dir, 'good')
            self.filenames = [os.path.join(dir, filename) for filename in os.listdir(dir)]

        elif type == DatasetType.Test:
            self.filenames = []
            self.labels = []
            self.gt = []
            for label in cfg.labels:
                dir = os.path.join(cfg.test_dir, label)
                listdir = os.listdir(dir)
                self.filenames = self.filenames + [os.path.join(dir, filename) for filename in listdir]

                self.labels = self.labels + [label] * len(listdir)

                if label == 'good':
                    self.gt = self.gt + [None] * len(listdir)
                else:
                    dir = os.path.join(cfg.gt_dir, label)
                    self.gt = self.gt + [os.path.join(dir, filename) for filename in os.listdir(dir)]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, item):

        if self.type == DatasetType.Train:
            image = self.transform(Image.open(self.filenames[item]).convert('RGB'))
            return image

        elif self.type == DatasetType.Test:
            image = self.transform(Image.open(self.filenames[item]).convert('RGB'))
            label = self.labels[item]
            gt = Image.open(self.gt[item]).convert('RGB') if label != 'good' else None
            return image, label, gt
