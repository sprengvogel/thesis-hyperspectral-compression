import torch.nn as nn

from compressai.models.google import JointAutoregressiveHierarchicalPriors


class GeneralHyperprior(JointAutoregressiveHierarchicalPriors):
    """General hyperprior model with main and hyperprior model given as input
    """

    def __init__(self, main_autoencoder, hyperprior_autoencoder, **kwargs):
        super().__init__(N=13, M=13, **kwargs)

        self.g_a = main_autoencoder.encoder

        self.h_a = hyperprior_autoencoder.encoder

        self.h_s = hyperprior_autoencoder.decoder

        self.g_s = main_autoencoder.decoder

    def forward(self, x):
        output = super().forward(x)
        return output["x_hat"], output["likelihoods"]["y"], output["likelihoods"]["z"]

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.conv1.weight"].size(0)
        net = cls(N)
        net.load_state_dict(state_dict)
        return net
