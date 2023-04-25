import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from hypercomp import metrics
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import random_split
import math
import numpy as np

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.VAE1DModel(
        nChannels=p.CHANNELS, latent_dim=26), lr=p.LR, loss=metrics.VAELoss(), model_type=models.ModelType.VAE)
    summary(model.autoencoder, input_size=(p.BATCH_SIZE,
            p.CHANNELS, 128, 128), device="cuda:"+str(p.GPU_ID))

    data.train_and_test(model)