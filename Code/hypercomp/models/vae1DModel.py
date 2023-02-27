import torch
import torch.nn as nn
import numpy as np
from .. import params as p
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class VAE1DModel(torch.nn.Module):

    def __init__(self, nChannels: int, latent_dim) -> None:
        super().__init__()
        self.encoder = VAE1DEncoder(
            input_channels=nChannels, latent_dim=latent_dim)
        self.decoder = VAE1DDecoder(
            output_channels=nChannels, latent_dim=latent_dim)

    def forward(self, x):
        mean, log_variance, z = self.encoder(x)
        return self.decoder(z), mean, log_variance


class VAE1DEncoder(torch.nn.Module):

    def __init__(self, input_channels: int, latent_dim: int) -> None:
        super().__init__()
        # Round up to next number divisible by 4
        self.input_channels = input_channels
        divisor = 8
        self.padded_channels = input_channels + paddingToBeDivisibleByN(
            input_size=input_channels, divisor=divisor)
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=self.padded_channels //
                      2, kernel_size=11, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=self.padded_channels//2, out_channels=self.padded_channels//4, kernel_size=11, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=self.padded_channels//4, out_channels=self.padded_channels//8, kernel_size=11, padding='same'),
            nn.LeakyReLU(),
            nn.MaxPool1d(kernel_size=2),
            nn.Conv1d(
                in_channels=self.padded_channels//8, out_channels=self.padded_channels//16, kernel_size=9, padding='same'),
            nn.LeakyReLU(),
            nn.Conv1d(
                in_channels=self.padded_channels//16, out_channels=1, kernel_size=7, padding='same'),
            nn.LeakyReLU(),
            # nn.Flatten()
        )
        self.linear_mean = nn.Linear(self.padded_channels//8, latent_dim)
        self.linear_variance = nn.Linear(self.padded_channels//8, latent_dim)
        self.tanh = nn.Tanh()

    def forward(self, x: torch.Tensor):
        x = flatten_spacial_dims(x)
        if not x.dim() == 3:
            raise ValueError(
                """Input is expected in format (Batch, 1, Channels). Spatial dimensions should be flattened into the batch dimension.\n
                Shape was instead: """ + str(x.shape))
        out = torch.nn.functional.pad(
            x, pad=(0, self.padded_channels-self.input_channels), value=0)

        out = self.encoder(out)
        mean = self.tanh(self.linear_mean(out))
        log_variance = nn.functional.softplus(
            self.tanh(self.linear_variance(out)))
        z = self._sample_latent(mean, log_variance)
        return [mean, log_variance, z]

    def _sample_latent(self, mean, log_variance):
        std = torch.exp(log_variance / 2)
        q = torch.distributions.Normal(mean, std)
        z = q.rsample()
        return z


class VAE1DDecoder(torch.nn.Module):

    def __init__(self, output_channels: int, latent_dim: int) -> None:
        super().__init__()
        self.output_channels = output_channels
        divisor = 8
        self.padded_channels = output_channels + \
            paddingToBeDivisibleByN(output_channels, divisor=divisor)
        self.decoder = nn.Sequential(
            nn.Linear(latent_dim, self.padded_channels//8),
            nn.LeakyReLU(),
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
            nn.Sigmoid()
        )

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
