import torch
import torch.nn as nn
import math
from compressai.models.waseda import Cheng2020Attention
from .. import params as p


class WengAttentionModel(nn.Module):

    def __init__(self, N: int = 192) -> None:
        super().__init__()
        self.model = Cheng2020Attention(N=N)

    def forward(self, x):
        output = self.model(x)
        return output["x_hat"], output["likelihoods"]["y"], output["likelihoods"]["z"]
