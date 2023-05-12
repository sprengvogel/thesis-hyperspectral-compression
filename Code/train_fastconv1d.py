import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p

import torch
from torch.utils.data import random_split
import math
import numpy as np

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Fast1DConvModel(
        nChannels=202, H=128, W=128, bottleneck_size=1), lr=p.LR)
    data.train_and_test(model, batch_size=4)
