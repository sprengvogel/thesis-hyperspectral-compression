import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import random_split
import math
import numpy as np

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    model = models.LitAutoEncoder(models.Conv1DModel(
        nChannels=p.CHANNELS, bpp_2=True), lr=p.LR)
    model.load_from_checkpoint(
        "MastersThesis/1oj2tot0/checkpoints/last.ckpt", model=model.autoencoder)
    data.train_and_test(model)
