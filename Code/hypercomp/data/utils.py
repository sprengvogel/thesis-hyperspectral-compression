import numpy as np
from torch.utils.data import Dataset, DataLoader
from .. import params as p


def dataLoader(dataset: Dataset, batch_size = p.BATCH_SIZE) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=p.SHUFFLE_DATA_LOADER, num_workers=p.NUM_WORKERS, pin_memory=True, drop_last=True)
