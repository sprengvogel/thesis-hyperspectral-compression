import torch
from .conv1DModel import Conv1DEncoder, Conv1DDecoder
from .conv2DModel import Conv2DModel
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class CombinedModel(torch.nn.Module):
    def __init__(self, nChannels: int, innerChannels: int, H: int = 96, W: int = 96) -> None:
        super().__init__()
        self.outer_encoder = Conv1DEncoder(input_channels=nChannels)
        self.outer_decoder = Conv1DDecoder(output_channels=nChannels)
        self.inner_autoencoder = Conv2DModel(nChannels=innerChannels, H=H, W=W)

    def forward(self, x):
        latent_image = unflatten_and_split_apart_batches(self.outer_encoder(x))
        x_hat_inner = self.inner_autoencoder(latent_image)
        x_hat = self.outer_decoder(flatten_spacial_dims(x_hat_inner))
        return x_hat, x_hat_inner, latent_image
