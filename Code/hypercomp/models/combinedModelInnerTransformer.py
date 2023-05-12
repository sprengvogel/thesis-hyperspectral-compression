import torch
from .conv1DModel import Conv1DModel, Conv1DEncoder, Conv1DDecoder
from .visionTransformer import VisionTransformer
from .utils import unflatten_and_split_apart_batches, flatten_spacial_dims


class CombinedModelInnerTransformer(torch.nn.Module):
    def __init__(self, nChannels: int, innerChannels: int, outerModel: Conv1DModel = None) -> None:
        super().__init__()
        if outerModel == None:
            self.outer_encoder = Conv1DEncoder(
                input_channels=nChannels, num_poolings=4)
            self.outer_decoder = Conv1DDecoder(
                output_channels=nChannels, num_poolings=4)
        else:
            self.outer_encoder = outerModel.encoder
            self.outer_decoder = outerModel.decoder
        self.inner_autoencoder = VisionTransformer(embed_dim=101,
                                                   hidden_dim=202,
                                                   num_channels=innerChannels,
                                                   patch_size=8,
                                                   num_heads=8,
                                                   num_layers=6,
                                                   num_patches=64,
                                                   dropout=0.2)

    def forward(self, x):
        latent_image = unflatten_and_split_apart_batches(self.outer_encoder(x))
        x_hat_inner, y_likelihoods, z_likelihoods = self.inner_autoencoder(
            latent_image)
        x_hat = self.outer_decoder(flatten_spacial_dims(x_hat_inner))
        return x_hat, x_hat_inner, latent_image, y_likelihoods, z_likelihoods
