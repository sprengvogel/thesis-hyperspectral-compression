import csv
import os

import numpy as np
import torch

from torch.utils.data import Dataset


class HySpecNet11k(Dataset):
    """
    Dataset:
        HySpecNet11k
    Authors:
        Martin Hermann Paul Fuchs
        Begüm Demir
    Paper:
        <Link To Paper>
    Cite:
        <Bibtex Citation>

    Folder Structure:
        - root_dir/
            - patches/
                - tile_001
                    - tile_001-patch_01
                        - tile_001-patch_01-DATA.npy
                        - tile_001-patch_01-QL_PIXELMASK.TIF
                        - tile_001-patch_01-QL_QUALITY_CIRRUS.TIF
                        - tile_001-patch_01-QL_QUALITY_CLASSES.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUD.TIF
                        - tile_001-patch_01-QL_QUALITY_CLOUDSHADOW.TIF
                        - tile_001-patch_01-QL_QUALITY_HAZE.TIF
                        - tile_001-patch_01-QL_QUALITY_SNOW.TIF
                        - tile_001-patch_01-QL_QUALITY_TESTFLAGS.TIF
                        - tile_001-patch_01-QL_SWIR.TIF
                        - tile_001-patch_01-QL_VNIR.TIF
                        - tile_001-patch_01-SPECTRAL_IMAGE.TIF
                        - tile_001-patch_01-THUMBNAIL.jpg
                    - tile_001-patch_02
                        - ...
                    - ...
                - tile_002
                    - ...
                - ...
            - splits/
                - easy/
                    - test.csv
                    - train.csv
                    - val.csv
                - hard/
                    - test.csv
                    - train.csv
                    - val.csv
                - ...
            - ...
    """

    def __init__(self, root_dir, mode="easy", split="train", transform=None):
        self.root_dir = root_dir

        self.csv_path = os.path.join(
            self.root_dir, "splits", mode, f"{split}.csv")
        with open(self.csv_path, newline='') as f:
            csv_reader = csv.reader(f)
            csv_data = list(csv_reader)
            self.npy_paths = sum(csv_data, [])
        self.npy_paths = [os.path.join(
            self.root_dir, "patches", x) for x in self.npy_paths]

        self.transform = transform

    def __len__(self):
        return len(self.npy_paths)

    def __getitem__(self, index):
        # get full numpy path
        npy_path = self.npy_paths[index]
        # read numpy data
        img = np.load(npy_path)
        # convert numpy array to pytorch tensor
        img = torch.from_numpy(img)
        # apply transformations
        if self.transform:
            img = self.transform(img)
        return img
