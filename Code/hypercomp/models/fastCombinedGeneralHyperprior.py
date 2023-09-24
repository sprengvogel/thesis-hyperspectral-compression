import torch
from .fast1DconvModel import Fast1DConvModel, Fast1DConvEncoder, Fast1DConvDecoder


class FastCombinedGeneralHyperprior(torch.nn.Module):
    def __init__(self, nChannels: int, bottleneck_size: int, H: int = 128, W: int = 128, outerModel: Fast1DConvModel = None, innerModel=None) -> None:
        super().__init__()
        if outerModel == None:
            self.outer_encoder = Fast1DConvEncoder(
                input_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)
            self.outer_decoder = Fast1DConvDecoder(
                output_channels=nChannels, H=H, W=W, bottleneck_size=bottleneck_size)
        else:
            self.outer_encoder = outerModel.encoder
            self.outer_decoder = outerModel.decoder
        self.inner_autoencoder = innerModel

    def forward(self, x):
        latent_image = self.outer_encoder(x)
        x_hat_inner, y_likelihoods, z_likelihoods = self.inner_autoencoder(
            latent_image)
        x_hat = self.outer_decoder(x_hat_inner)
        return x_hat, x_hat_inner, latent_image, y_likelihoods, z_likelihoods
