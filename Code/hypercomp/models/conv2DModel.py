import torch
import torch.nn as nn
import math
from .. import params as p


class Conv2DModel(nn.Module):

    def __init__(self, nChannels: int, H: int, W: int, kernel_size: int = 3, use_groups: bool = False) -> None:
        super().__init__()
        self.encoder = Conv2DEncoder(
            input_channels=nChannels, H=H, W=W, kernel_size=kernel_size, use_groups=use_groups)
        self.decoder = Conv2DDecoder(
            output_channels=nChannels, H=H, W=W, kernel_size=kernel_size, use_groups=use_groups)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Conv2DEncoder(nn.Module):

    def __init__(self, input_channels: int, H: int, W: int, kernel_size: int, use_groups: bool) -> None:
        super().__init__()
        if use_groups:
            groups = input_channels
            small_filter_nr = 32*groups
            big_filter_nr = 64*groups
        else:
            groups = 1
            small_filter_nr = 256
            big_filter_nr = 512
        padding = (kernel_size-1)//2
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=small_filter_nr, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((small_filter_nr, H, W)),
            nn.PReLU(small_filter_nr),
            nn.Conv2d(in_channels=small_filter_nr,
                      out_channels=small_filter_nr, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((small_filter_nr, H, W)),
            nn.PReLU(small_filter_nr),
            nn.Conv2d(in_channels=small_filter_nr, out_channels=big_filter_nr,
                      kernel_size=2, stride=2, groups=groups),
            nn.LayerNorm((big_filter_nr, H//2, W//2)),
            nn.PReLU(big_filter_nr),
            nn.Conv2d(in_channels=big_filter_nr, out_channels=small_filter_nr,
                      kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((small_filter_nr, H//2, W//2)),
            nn.PReLU(small_filter_nr),
            nn.Conv2d(
                in_channels=small_filter_nr, out_channels=input_channels, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.Sigmoid())  # ,
        # nn.PReLU(512) ,
        # nn.MaxPool2d(2, 2))  # ,
        # nn.Conv2d(
        #    in_channels=512, out_channels=512, kernel_size=3, padding=1),
        # nn.PReLU(512),
        # nn.MaxPool2d(2, 2))  # ,
        # nn.Flatten(),
        # nn.Linear(512*H*W, 128*H*W))

    def forward(self, x):
        return self.encoder(x)


class Conv2DDecoder(nn.Module):

    def __init__(self, output_channels: int, H: int, W: int, kernel_size: int, use_groups: bool) -> None:
        super().__init__()
        self.H, self.W = H, W
        if use_groups:
            groups = output_channels
            small_filter_nr = 32*groups
            big_filter_nr = 64*groups
        else:
            groups = 1
            small_filter_nr = 256
            big_filter_nr = 512
        padding = (kernel_size-1)//2
        # self.linear = nn.Linear(128*H*W, 512*H*W)
        self.decoder = nn.Sequential(
            # nn.PReLU(512*H*W),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # nn.PReLU(512),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # nn.PReLU(output_channels),
            nn.ConvTranspose2d(output_channels, small_filter_nr,
                               kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((small_filter_nr, H//2, W//2)),
            nn.PReLU(small_filter_nr),
            nn.ConvTranspose2d(
                small_filter_nr, big_filter_nr, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((big_filter_nr, H//2, W//2)),
            nn.PReLU(big_filter_nr),
            nn.ConvTranspose2d(big_filter_nr, small_filter_nr, kernel_size=2,
                               stride=2, groups=groups),
            nn.LayerNorm((small_filter_nr, H, W)),
            nn.PReLU(small_filter_nr),
            nn.ConvTranspose2d(
                small_filter_nr, small_filter_nr, kernel_size=kernel_size, padding=padding, groups=groups),
            nn.LayerNorm((small_filter_nr, H, W)),
            nn.PReLU(small_filter_nr),
            nn.ConvTranspose2d(small_filter_nr, output_channels,
                               kernel_size=kernel_size, padding=padding, groups=groups),
            nn.Sigmoid())

    def forward(self, x):
        # out = self.linear(x)
        # out = out.reshape((x.shape[0], 512, self.H, self.W))
        out = self.decoder(x)
        return out
