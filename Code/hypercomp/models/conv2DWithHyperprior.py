import math
import torch
import torch.nn as nn
from compressai.entropy_models import GaussianConditional
from .utils import update_registered_buffers, conv, deconv
from .entropyBottleneckCompressionModel import EntropyBottleneckCompressionModel


class Conv2DWithHyperprior(EntropyBottleneckCompressionModel):
    r"""Scale Hyperprior model from J. Balle, D. Minnen, S. Singh, S.J. Hwang,
    N. Johnston: `"Variational Image Compression with a Scale Hyperprior"
    <https://arxiv.org/abs/1802.01436>`_ Int. Conf. on Learning Representations
    (ICLR), 2018.
    Args:
        N (int): Number of channels
        M (int): Number of channels in the expansion layers (last layer of the
            encoder and last layer of the hyperprior decoder)
    """

    def __init__(self, input_channels=26, H=128, W=128, N=64, M=101, **kwargs):
        super().__init__(entropy_bottleneck_channels=N, **kwargs)

        self.g_a = nn.Sequential(
            nn.Conv2d(in_channels=input_channels,
                      out_channels=256, kernel_size=3, padding=1),
            nn.LayerNorm((256, H, W)),
            nn.PReLU(256),
            nn.MaxPool2d(2, 2),
            nn.Conv2d(in_channels=256,
                      out_channels=256, kernel_size=3, padding=1),
            nn.LayerNorm((256, H//2, W//2)),
            nn.PReLU(256),
            #nn.MaxPool2d(2, 2),
            nn.Conv2d(
                in_channels=256, out_channels=M, kernel_size=3, padding=1),
            nn.Sigmoid())

        self.g_s = nn.Sequential(
            # nn.PReLU(512*H*W),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # nn.PReLU(512),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.PReLU(M),
            nn.ConvTranspose2d(M, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((256, H//2, W//2)),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.LayerNorm((256, H, W)),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, input_channels, kernel_size=3,
                               stride=1, padding=1),
            nn.Sigmoid())

        self.h_a = nn.Sequential(
            conv(M, N, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
            conv(N, N, stride=1),
            nn.ReLU(inplace=True),
            conv(N, N),
        )

        self.h_s = nn.Sequential(
            deconv(N, N),
            nn.ReLU(inplace=True),
            deconv(N, N, stride=1),
            nn.ReLU(inplace=True),
            conv(N, M, stride=1, kernel_size=3),
            nn.ReLU(inplace=True),
        )

        self.gaussian_conditional = GaussianConditional(None)
        self.N = int(N)
        self.M = int(M)

    @property
    def downsampling_factor(self) -> int:
        return 2 ** (4 + 2)

    def forward(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        scales_hat = self.h_s(z_hat)
        y_hat, y_likelihoods = self.gaussian_conditional(y, scales_hat)
        x_hat = self.g_s(y_hat)

        return x_hat, y_likelihoods, z_likelihoods

    def load_state_dict(self, state_dict):
        update_registered_buffers(
            self.gaussian_conditional,
            "gaussian_conditional",
            ["_quantized_cdf", "_offset", "_cdf_length", "scale_table"],
            state_dict,
        )
        super().load_state_dict(state_dict)

    @classmethod
    def from_state_dict(cls, state_dict):
        """Return a new model instance from `state_dict`."""
        N = state_dict["g_a.0.weight"].size(0)
        M = state_dict["g_a.6.weight"].size(0)
        net = cls(N, M)
        net.load_state_dict(state_dict)
        return net

    def update(self, scale_table=None, force=False):
        if scale_table is None:
            scale_table = get_scale_table()
        updated = self.gaussian_conditional.update_scale_table(
            scale_table, force=force)
        updated |= super().update(force=force)
        return updated

    def compress(self, x):
        y = self.g_a(x)
        z = self.h_a(torch.abs(y))

        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_strings = self.gaussian_conditional.compress(y, indexes)
        return {"strings": [y_strings, z_strings], "shape": z.size()[-2:]}

    def decompress(self, strings, shape):
        assert isinstance(strings, list) and len(strings) == 2
        z_hat = self.entropy_bottleneck.decompress(strings[1], shape)
        scales_hat = self.h_s(z_hat)
        indexes = self.gaussian_conditional.build_indexes(scales_hat)
        y_hat = self.gaussian_conditional.decompress(
            strings[0], indexes, z_hat.dtype)
        x_hat = self.g_s(y_hat).clamp_(0, 1)
        return x_hat


# From Balle's tensorflow compression examples
SCALES_MIN = 0.11
SCALES_MAX = 256
SCALES_LEVELS = 64


def get_scale_table(min=SCALES_MIN, max=SCALES_MAX, levels=SCALES_LEVELS):
    return torch.exp(torch.linspace(math.log(min), math.log(max), levels))
