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
from hypercomp import metrics

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv2DModel(
        nChannels=p.CHANNELS, H=128, W=128), lr=p.LR)
    data.train_and_test(model, batch_size=p.BATCH_SIZE_CONV2D)
