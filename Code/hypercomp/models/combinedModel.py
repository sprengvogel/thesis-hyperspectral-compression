import torch
from . import Conv1DEncoder, Conv1DDecoder, ScaleHyperprior
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class CombinedModel(torch.nn.Module):
    def __init__(self, nChannels: int, hyperpriorChannels: int, N: int = 128, M: int = 192) -> None:
        super().__init__()
        self.conv_encoder = Conv1DEncoder(input_channels=nChannels)
        self.conv_decoder = Conv1DDecoder(output_channels=nChannels)
        self.hyperprior = ScaleHyperprior(
            channel_number=hyperpriorChannels, N=N, M=M)

    def forward(self, x):
        latent_image = unflatten_and_split_apart_batches(self.conv_encoder(x))
        x_hat, y_likelihoods, z_likelihoods = self.hyperprior(latent_image)
        x_hat = flatten_spacial_dims(x_hat)
        return self.conv_decoder(x_hat), y_likelihoods, z_likelihoods
