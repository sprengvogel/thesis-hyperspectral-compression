import torch
from .conv1DModel import Conv1DModel, Conv1DEncoder, Conv1DDecoder
from .conv2DModel import Conv2DModel
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class CombinedModel(torch.nn.Module):
    def __init__(self, nChannels: int, innerChannels: int, H: int = 128, W: int = 128, outerModel: Conv1DModel = None, innerModel=None) -> None:
        super().__init__()
        if outerModel == None:
            self.outer_encoder = Conv1DEncoder(
                input_channels=nChannels, num_poolings=4)
            self.outer_decoder = Conv1DDecoder(
                output_channels=nChannels, num_poolings=4)
        else:
            self.outer_encoder = outerModel.encoder
            self.outer_decoder = outerModel.decoder
        if innerModel == None:
            self.inner_autoencoder = Conv2DModel(
                nChannels=innerChannels, H=H, W=W)
        else:
            self.inner_autoencoder = innerModel

    def forward(self, x):
        latent_image = unflatten_and_split_apart_batches(self.outer_encoder(x))
        x_hat_inner = self.inner_autoencoder(latent_image)
        x_hat = self.outer_decoder(flatten_spacial_dims(x_hat_inner))
        return x_hat, x_hat_inner, latent_image

    def encode(self, x):
        latent_image = unflatten_and_split_apart_batches(self.outer_encoder(x))
        inner_latent = self.inner_autoencoder.encoder(latent_image)
        return inner_latent
