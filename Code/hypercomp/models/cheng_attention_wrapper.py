import torch.nn as nn
from .cheng_attention import Cheng2020Attention


class ChengAttentionModel(nn.Module):

    def __init__(self, in_channels, N: int = 192) -> None:
        super().__init__()
        self.model = Cheng2020Attention(in_channels=in_channels, N=N)

    def forward(self, x):
        output = self.model(x)
        return output["x_hat"], output["likelihoods"]["y"], output["likelihoods"]["z"]

    def compress(self, x):
        return self.model.compress(x)

    def decompress(self, strings, shape):
        return self.model.decompress(strings, shape)

    def update(self):
        return self.model.update()