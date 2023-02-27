import torch
import torch.nn as nn
import math
from .. import params as p
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class Conv1DModel(torch.nn.Module):

    def __init__(self, nChannels: int, bpp_2: bool = False) -> None:
        super().__init__()
        self.encoder = Conv1DEncoder(input_channels=nChannels, bpp_2=bpp_2)
        self.decoder = Conv1DDecoder(output_channels=nChannels, bpp_2=bpp_2)

    def forward(self, x):
        return self.decoder(self.encoder(x))


class Conv1DEncoder(torch.nn.Module):

    def __init__(self, input_channels: int, bpp_2: bool = False) -> None:
        super().__init__()
        # Round up to next number divisible by 4
        self.input_channels = input_channels
        self.bpp_2 = bpp_2
        if bpp_2:
            divisor = 16
        else:
            divisor = 8
        self.padded_channels = input_channels + paddingToBeDivisibleByN(
            input_size=input_channels, divisor=divisor)
        print(self.input_channels)
        print(self.padded_channels)
        if bpp_2:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=self.padded_channels //
                          2, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//2, out_channels=self.padded_channels//4, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//4, out_channels=self.padded_channels//8, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//8, out_channels=self.padded_channels//16, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//16, out_channels=self.padded_channels//32, kernel_size=9, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.Conv1d(
                    in_channels=self.padded_channels//32, out_channels=1, kernel_size=7, padding='same'),
                nn.Sigmoid())
        else:
            self.encoder = nn.Sequential(
                nn.Conv1d(in_channels=1, out_channels=self.padded_channels //
                          2, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//2, out_channels=self.padded_channels//4, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//4, out_channels=self.padded_channels//8, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.MaxPool1d(kernel_size=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//8, out_channels=self.padded_channels//16, kernel_size=9, padding='same'),
                nn.LeakyReLU(),
                # nn.Dropout1d(p=0.1),
                nn.Conv1d(
                    in_channels=self.padded_channels//16, out_channels=1, kernel_size=7, padding='same'),
                nn.Sigmoid()
            )

    def forward(self, x: torch.Tensor):
        x = flatten_spacial_dims(x)
        if not x.dim() == 3:
            raise ValueError(
                """Input is expected in format (Batch, 1, Channels). Spatial dimensions should be flattened into the batch dimension.\n
                Shape was instead: """ + str(x.shape))
        out = torch.nn.functional.pad(
            x, pad=(0, self.padded_channels-self.input_channels), value=0)
        return self.encoder(out)


class Conv1DDecoder(torch.nn.Module):

    def __init__(self, output_channels: int, bpp_2: bool = False) -> None:
        super().__init__()
        self.output_channels = output_channels
        self.bpp_2 = bpp_2
        if bpp_2:
            divisor = 16
        else:
            divisor = 8
        self.padded_channels = output_channels + \
            paddingToBeDivisibleByN(output_channels, divisor=divisor)
        if self.bpp_2:
            self.decoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=1, out_channels=self.padded_channels//32, kernel_size=7, padding='same'),
                nn.LeakyReLU(),
                nn.Conv1d(
                    in_channels=self.padded_channels//32, out_channels=self.padded_channels//16, kernel_size=9, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//16, out_channels=self.padded_channels//8, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//8, out_channels=self.padded_channels//4, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//4, out_channels=self.padded_channels//2, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//2, out_channels=1, kernel_size=11, padding='same'),
                nn.Sigmoid())
        else:
            self.decoder = nn.Sequential(
                nn.Conv1d(
                    in_channels=1, out_channels=self.padded_channels//16, kernel_size=7, padding='same'),
                nn.LeakyReLU(),
                nn.Conv1d(
                    in_channels=self.padded_channels//16, out_channels=self.padded_channels//8, kernel_size=9, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//8, out_channels=self.padded_channels//4, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//4, out_channels=self.padded_channels//2, kernel_size=11, padding='same'),
                nn.LeakyReLU(),
                nn.Upsample(scale_factor=2),
                nn.Conv1d(
                    in_channels=self.padded_channels//2, out_channels=1, kernel_size=11, padding='same'),
                nn.Sigmoid())

    def forward(self, x: torch.Tensor):
        if not x.dim() == 3 and x.shape[1] == 1:
            raise ValueError(
                """Input is expected in format (Batch, 1, Channels). Spatial dimensions should be flattened into the batch dimension.\n
                Shape was instead: """ + str(x.shape))
        out = self.decoder(x)
        # Remove padding channels
        out = out[:, :, :self.output_channels]
        return unflatten_and_split_apart_batches(out)


def paddingToBeDivisibleByN(input_size: int, divisor: int) -> int:
    """
    Returns number of padding necessary to reach a number divisible by @divisor.
    """
    if input_size % divisor == 0:
        return 0
    return divisor - input_size % divisor
