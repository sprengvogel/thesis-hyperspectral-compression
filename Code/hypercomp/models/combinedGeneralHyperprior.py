import torch
from .conv1DModel import Conv1DModel, Conv1DEncoder, Conv1DDecoder
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class CombinedGeneralHyperprior(torch.nn.Module):
    def __init__(self, nChannels: int, outerModel: Conv1DModel = None, innerModel=None) -> None:
        super().__init__()
        if outerModel == None:
            self.outer_encoder = Conv1DEncoder(
                input_channels=nChannels, num_poolings=4)
            self.outer_decoder = Conv1DDecoder(
                output_channels=nChannels, num_poolings=4)
        else:
            self.outer_encoder = outerModel.encoder
            self.outer_decoder = outerModel.decoder
        self.inner_autoencoder = innerModel

    def forward(self, x):
        latent_image = unflatten_and_split_apart_batches(self.outer_encoder(x))
        x_hat_inner, y_likelihoods, z_likelihoods = self.inner_autoencoder(
            latent_image)
        x_hat = self.outer_decoder(flatten_spacial_dims(x_hat_inner))
        return x_hat, x_hat_inner, latent_image, y_likelihoods, z_likelihoods

    def compress(self, x):
        x_inner = unflatten_and_split_apart_batches(self.outer_encoder(x))
        return self.inner_autoencoder.compress(x_inner)

    def decompress(self, strings, shape):
        x_hat_inner = self.inner_autoencoder.decompress(strings, shape)[
            "x_hat"]
        return self.outer_decoder(flatten_spacial_dims(x_hat_inner))

    def update(self):
        return self.inner_autoencoder.update()
