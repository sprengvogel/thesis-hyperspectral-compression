from pydantic import validate_arguments, FilePath
from scipy.io import loadmat
from pathlib import Path
import numpy as np
from typing import List, Any
import random
from pydantic import validate_arguments, DirectoryPath, confloat, BaseModel, FilePath
import math
from enum import Enum

class Split(str, Enum):
    train = "train"
    validation = "validation"
    test = "test"

    def __str__(self):
        return self.value

class DataSplit(BaseModel):
    train: List[Any]
    validation: List[Any]
    test: List[Any]

@validate_arguments
def get_mat_paths(
    mat_dataset_path: DirectoryPath,
    glob_pattern: str = "*.mat",
) -> List[Path]:
    mat_paths = list(mat_dataset_path.glob(glob_pattern))
    mat_paths.sort()
    return mat_paths

@validate_arguments
def random_data_split(
    data: List[Any],
    train_split_perc: confloat(gt=0.0, lt=1.0),
    validation_split_perc: confloat(gt=0.0, lt=1.0),
    test_split_perc: confloat(gt=0.0, lt=1.0),
    shuffle: bool = True,
    seed: int = 42
) -> DataSplit:
    """
    Split data according to percentages.

    This function guarantees that validation and test splits have at least one element by using
    `math.ceil` after calculating the fraction.

    May shuffle the data.

    Returns the entire data in memory as a `DataSplit` element
    """
    overall_perc = train_split_perc + validation_split_perc + test_split_perc
    if not np.isclose(overall_perc, 1.0):
        raise ValueError("Split percentages have to add up to 1.0!")

    if shuffle:
        random.seed(seed)
        random.shuffle(data)

    data_length = len(data)
    # guarantee that val/test have at least 1 element
    val_length = math.ceil(data_length * validation_split_perc)
    test_length = math.ceil(data_length * test_split_perc)
    train_length = data_length - val_length - test_length

    if train_length <= 0:
        raise RuntimeError("The resulting train length would be smaller than 0! Please adjust your percentages!")

    return DataSplit(
        train=data[:train_length],
        validation=data[train_length:train_length + val_length],
        test=data[train_length + val_length:]
    )

@validate_arguments
def mat_path_to_np(mat_path: FilePath, array_key: str):
    # could potentially grow more complex
    return loadmat(mat_path)[array_key]

def np_to_torch_tns_layout(arr):
    """
    Convert a (H x W x C) numpy array to a (C x H x W) array.
    Move the `numpy` ordering to the default PyTorch Tensor ordering.
    """
    if arr.ndim != 3:
        raise ValueError("Function is only defined for 3 dimensional input!")
    return arr.transpose((2, 0, 1))
