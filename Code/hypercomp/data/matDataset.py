import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import os
import numpy as np
from .utils import dataLoader


class MatDataset(Dataset):
    """
    Dataset reading all .mat files from a single directory.
    All files are read on init, so that __getitem__ is as fast as possible.
    Returns shape (channels, x_dim, y_dim)
    """

    def __init__(self, source_dir, transform=None):
        self.mats = []
        # Read all .mat files in the source directory
        for file in os.listdir(source_dir):
            if file.endswith(".mat"):
                file_path = os.path.join(source_dir, file)
                mat = scipy.io.loadmat(file_path)['img']
                # We want the shape (369,96,96) instead of (96,96,369)
                self.mats.append(np.moveaxis(mat, -1, 0))
        self.transform = transform

    def __len__(self):
        return len(self.mats)

    def __getitem__(self, index):
        if self.transform:
            return self.transform(self.mats(index))
        return self.mats[index]


if __name__ == "__main__":
    dataset = MatDataset("hypercomp/data/mat-data/")
    dataloader = dataLoader(dataset)
    for el in dataloader:
        print(el)
        print(el.shape)
