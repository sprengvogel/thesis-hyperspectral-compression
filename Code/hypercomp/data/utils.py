from torch.utils.data import Dataset, DataLoader
from .. import params as p


def dataLoader(dataset: Dataset) -> DataLoader:
    return DataLoader(dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE_DATA_LOADER, num_workers=p.NUM_WORKERS)
