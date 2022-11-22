import torch
from torch.utils.data import Dataset, DataLoader
import scipy.io
import os
import numpy as np
from .utils import dataLoader
from tqdm import tqdm


class MatDataset(Dataset):
    """
    Dataset reading all .mat files from a single directory.
    All files are read on init, so that __getitem__ is as fast as possible.
    Returns shape (batch_size, channels, x_dim, y_dim)
    If spacial_flatten=True returns (batch_size, channels)
    """

    def __init__(self, source_dir, spacial_flatten=True, transform=None):
        if spacial_flatten:
            self.mats = None
        else:
            self.mats = []
        # Read all .mat files in the source directory
        for file in tqdm(os.listdir(source_dir)):
            if file.endswith(".mat"):
                file_path = os.path.join(source_dir, file)
                mat = scipy.io.loadmat(file_path)['img']
                if spacial_flatten:
                    if self.mats is None:
                        self.mats = np.empty((0, mat.shape[-1]))
                    # Transform shape from (96,96,369) to (96*96,369)
                    mat = mat.reshape(-1, mat.shape[-1])
                    # Extend mats array by the new pixels
                    self.mats = np.vstack((self.mats, mat))
                else:
                    # We want the shape (369,96,96) instead of (96,96,369)
                    self.mats.append(np.moveaxis(mat, -1, 0))
        self.mats = np.float32(np.expand_dims(self.mats, 1))
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
