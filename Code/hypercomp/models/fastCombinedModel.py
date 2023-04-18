import torch
from .fast1DconvModel import Fast1DConvModel, Fast1DConvEncoder, Fast1DConvDecoder
from .conv2DModel import Conv2DModel


class FastCombinedModel(torch.nn.Module):
    def __init__(self, nChannels: int, bottleneck_size: int, H: int = 96, W: int = 96, outerModel: Fast1DConvModel = None) -> None:
        super().__init__()
        if outerModel == None:
            self.outer_encoder = Fast1DConvEncoder(
                input_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)
            self.outer_decoder = Fast1DConvDecoder(
                output_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)
        else:
            self.outer_encoder = outerModel.encoder
            self.outer_decoder = outerModel.decoder
        self.inner_autoencoder = Conv2DModel(
            nChannels=bottleneck_size, H=H, W=W)

    def forward(self, x):
        latent_image = self.outer_encoder(x)
        x_hat_inner = self.inner_autoencoder(latent_image)
        x_hat = self.outer_decoder(x_hat_inner)
        return x_hat, x_hat_inner, latent_image
