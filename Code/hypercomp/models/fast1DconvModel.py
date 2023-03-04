import torch
import torch.nn as nn
import math
from .. import params as p


class Fast1DConvModel(nn.Module):

    def __init__(self, nChannels: int, H: int, W: int, bottleneck_size: int) -> None:
        super().__init__()
        self.encoder = Fast1DConvEncoder(
            input_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)
        self.decoder = Fast1DConvDecoder(
            output_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Fast1DConvEncoder(nn.Module):

    def __init__(self, input_channels: int, H: int, W: int, bottleneck_size: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=256, kernel_size=1),
            nn.LayerNorm((H, W, 256)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=256, out_channels=512, kernel_size=1),
            nn.LayerNorm((H, W, 512)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=512, out_channels=1024, kernel_size=1),
            nn.LayerNorm((H, W, 1024)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024, out_channels=1024, kernel_size=1),
            nn.LayerNorm((H, W, 1024)),
            nn.LeakyReLU(),
            nn.Conv2d(in_channels=1024,
                      out_channels=bottleneck_size, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.encoder(x)


class Fast1DConvDecoder(nn.Module):

    def __init__(self, output_channels: int, H: int, W: int, bottleneck_size: int) -> None:
        super().__init__()
        self.H, self.W = H, W
        self.decoder = nn.Sequential(
            nn.Conv2d(bottleneck_size, 1024, kernel_size=1),
            nn.LayerNorm((H, W, 1024)),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 1024, kernel_size=1),
            nn.LayerNorm((H, W, 1024)),
            nn.LeakyReLU(),
            nn.Conv2d(1024, 512, kernel_size=1),
            nn.LayerNorm((H, W, 512)),
            nn.LeakyReLU(),
            nn.Conv2d(512, 256, kernel_size=1),
            nn.LayerNorm((H, W, 256)),
            nn.LeakyReLU(),
            nn.Conv2d(256, output_channels, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        out = self.decoder(x)
        return out
