import torch
import torch.nn as nn
import math
from .. import params as p


class Conv2DModel(nn.Module):

    def __init__(self, nChannels: int, H: int, W: int) -> None:
        super().__init__()
        self.encoder = Conv2DEncoder(input_channels=nChannels, H=H, W=W)
        self.decoder = Conv2DDecoder(output_channels=nChannels, H=H, W=W)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Conv2DEncoder(nn.Module):

    def __init__(self, input_channels: int, H: int, W: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256,
                      out_channels=256, kernel_size=3, padding=1),
            nn.PReLU(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels=256, out_channels=512, kernel_size=3, padding=1),
            nn.PReLU(512),
            nn.MaxPool2d(2, 2))  # ,
        # nn.Conv2d(
        #    in_channels=512, out_channels=512, kernel_size=3, padding=1),
        # nn.PReLU(512),
        # nn.MaxPool2d(2, 2))  # ,
        # nn.Flatten(),
        # nn.Linear(512*H*W, 128*H*W))

    def forward(self, x):
        return self.encoder(x)


class Conv2DDecoder(nn.Module):

    def __init__(self, output_channels: int, H: int, W: int) -> None:
        super().__init__()
        self.H, self.W = H, W
        #self.linear = nn.Linear(128*H*W, 512*H*W)
        self.decoder = nn.Sequential(
            # nn.PReLU(512*H*W),
            #nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # nn.PReLU(512),
            nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.PReLU(512),
            nn.ConvTranspose2d(512, 256, kernel_size=2, stride=2),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, output_channels, kernel_size=3,
                               stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        #out = self.linear(x)
        #out = out.reshape((x.shape[0], 512, self.H, self.W))
        out = self.decoder(x)
        return out
