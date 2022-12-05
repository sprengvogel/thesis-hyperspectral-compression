import numpy as np
from torch.utils.data import Dataset, DataLoader
from .. import params as p


def dataLoader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE_DATA_LOADER, num_workers=p.NUM_WORKERS, pin_memory=True, drop_last=True)


"""
Flattens a numpy array from dimensions (channels, height, width) to (height*width, channels)
"""


def flatten_spacial_dims(x):
    if not len(x.shape) == 3:
        raise ValueError(
            """Input is expected in format (Channels, Height, Width).\n
                Shape was instead: """ + str(x.shape))
    # Move channel dimension in the back
    x = np.moveaxis(x, 0, -1)
    # Then flatten front three dimension together
    return x.reshape(-1, x.shape[-1])
