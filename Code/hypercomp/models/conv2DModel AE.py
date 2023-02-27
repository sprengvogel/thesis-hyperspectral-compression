import torch
import torch.nn as nn
import math
from .. import params as p
import torchac


class STEQuantize(torch.autograd.Function):
    """Straight-Through Estimator for Quantization.
    Forward pass implements quantization by rounding to integers,
    backward pass is set to gradients of the identity function.
    """
    @staticmethod
    def forward(ctx, x):
        ctx.save_for_backward(x)
        return x.round()

    @staticmethod
    def backward(ctx, grad_outputs):
        return grad_outputs


class Conv2DModelAE(nn.Module):

    def __init__(self, nChannels: int, H: int, W: int) -> None:
        super().__init__()
        self.encoder = Conv2DEncoderAE(input_channels=nChannels, H=H, W=W)
        self.decoder = Conv2DDecoderAE(output_channels=nChannels, H=H, W=W)
        self.arithmeticEncoder = ConditionalProbabilityModel(
            p.LForAE, bottleneck_shape=(101, H//2, W//2))
        self.quantize = STEQuantize.apply

    def forward(self, x):
        latent = self.encoder(x)
        # The jiggle is there so that the lowest and highest symbol are not at
        # the boundary. Probably not needed.
        jiggle = 0.2
        spread = p.LForAE - 1 + jiggle
        # The sigmoid clamps to [0, 1], then we multiply it by spread and substract
        # spread / 2, so that the output is nicely centered around zero and
        # in the interval [-spread/2, spread/2]
        latent = torch.sigmoid(latent) * spread - spread / 2
        latent_quantized = self.quantize(latent)
        reconstructions = self.decoder(latent_quantized)
        sym = latent_quantized + p.LForAE // 2
        sym = sym.to(torch.long)
        bits_estimated, bits_real = self.arithmeticEncoder(
            sym.detach(), latent.detach())
        return reconstructions, sym, bits_estimated, bits_real


class Conv2DEncoderAE(nn.Module):

    def __init__(self, input_channels: int, H: int, W: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
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
                in_channels=256, out_channels=101, kernel_size=3, padding=1),
            nn.Sigmoid())  # ,
        # nn.PReLU(512) ,
        # nn.MaxPool2d(2, 2))  # ,
        # nn.Conv2d(
        #    in_channels=512, out_channels=512, kernel_size=3, padding=1),
        # nn.PReLU(512),
        # nn.MaxPool2d(2, 2))  # ,
        # nn.Flatten(),
        # nn.Linear(512*H*W, 128*H*W))

    def forward(self, x):
        return self.encoder(x)


class Conv2DDecoderAE(nn.Module):

    def __init__(self, output_channels: int, H: int, W: int) -> None:
        super().__init__()
        self.H, self.W = H, W
        # self.linear = nn.Linear(128*H*W, 512*H*W)
        self.decoder = nn.Sequential(
            # nn.PReLU(512*H*W),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            # nn.PReLU(512),
            # nn.ConvTranspose2d(512, 512, kernel_size=2, stride=2),
            nn.PReLU(101),
            nn.ConvTranspose2d(101, 256, kernel_size=3, stride=1, padding=1),
            nn.LayerNorm((256, H//2, W//2)),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, 256, kernel_size=2, stride=2),
            nn.LayerNorm((256, H, W)),
            nn.PReLU(256),
            nn.ConvTranspose2d(256, output_channels, kernel_size=3,
                               stride=1, padding=1),
            nn.Sigmoid())

    def forward(self, x):
        # out = self.linear(x)
        # out = out.reshape((x.shape[0], 512, self.H, self.W))
        out = self.decoder(x)
        return out


class ConditionalProbabilityModel(nn.Module):
    def __init__(self, L, bottleneck_shape):
        super(ConditionalProbabilityModel, self).__init__()
        self.L = L
        self.bottleneck_shape = bottleneck_shape

        self.bottleneck_size, _, _ = bottleneck_shape

        # We predict a value for each channel, for each level.
        num_output_channels = self.bottleneck_size * L

        self.model = nn.Sequential(
            nn.Conv2d(1, self.bottleneck_size, 3, padding=1),
            nn.BatchNorm2d(self.bottleneck_size),
            nn.ReLU(),
            nn.Conv2d(self.bottleneck_size,
                      self.bottleneck_size, 3, padding=1),
            nn.BatchNorm2d(self.bottleneck_size),
            nn.ReLU(),
            nn.Conv2d(self.bottleneck_size, num_output_channels, 1, padding=0),
        )

    def forward(self, sym, latent):
        batch_size = sym.shape[0]
        _, H, W = self.bottleneck_shape
        # Construct the input, which is just the label of the current number
        # at each spatial location.
        #bottleneck_shape_with_batch_dim = (batch_size, 1, H, W)
        # static_input = torch.ones(
        #    bottleneck_shape_with_batch_dim, dtype=torch.float32, device=sym.device)
        #dynamic_input = static_input * labels.reshape(-1, 1, 1, 1)
        # Divide by 9 and substract 0.5 to center the input around 0 and make
        # it be contained in [-0.5, 0.5].
        dynamic_input = latent
        #dynamic_input = dynamic_input / 9 - 0.5

        # Get the output of the CNN.
        output = self.model(dynamic_input)
        _, C, H, W = output.shape
        assert C == self.bottleneck_size * self.L

        # Reshape it such that the probability per symbol has it's own dimension.
        # output_reshaped has shape (B, C, L, H, W).
        output_reshaped = output.reshape(
            batch_size, self.bottleneck_size, self.L, H, W)
        # Take the softmax over that dimension to make this into a normalized
        # probability distribution.
        output_probabilities = torch.functional.softmax(output_reshaped, dim=2)
        # Permute the symbols dimension to the end, as expected by torchac.
        # output_probabilities has shape (B, C, H, W, L).
        output_probabilities = output_probabilities.permute(0, 1, 3, 4, 2)
        # Estimate the bitrate from the PMF.
        estimated_bits = estimate_bitrate_from_pmf(
            output_probabilities, sym=sym)
        # Convert to a torchac-compatible CDF.
        output_cdf = pmf_to_cdf(output_probabilities)
        # torchac expects sym as int16, see README for details.
        sym = sym.to(torch.int16)
        # torchac expects CDF and sym on CPU.
        output_cdf = output_cdf.detach().cpu()
        sym = sym.detach().cpu()
        # Get real bitrate from the byte_stream.
        byte_stream = torchac.encode_float_cdf(
            output_cdf, sym, check_input_bounds=True)
        real_bits = len(byte_stream) * 8
        """ if _WRITE_BITS:
            # Write to a file.
            with open('outfile.b', 'wb') as fout:
                fout.write(byte_stream)
            # Read from a file.
            with open('outfile.b', 'rb') as fin:
                byte_stream = f in.read()"""
        assert torchac.decode_float_cdf(output_cdf, byte_stream).equal(sym)
        return estimated_bits, real_bits


def pmf_to_cdf(pmf):
    cdf = pmf.cumsum(dim=-1)
    spatial_dimensions = pmf.shape[:-1] + (1,)
    zeros = torch.zeros(spatial_dimensions, dtype=pmf.dtype, device=pmf.device)
    cdf_with_0 = torch.cat([zeros, cdf], dim=-1)
    # On GPU, softmax followed by cumsum can lead to the final value being
    # slightly bigger than 1, so we clamp.
    cdf_with_0 = cdf_with_0.clamp(max=1.)
    return cdf_with_0


def estimate_bitrate_from_pmf(pmf, sym):
    L = pmf.shape[-1]
    pmf = pmf.reshape(-1, L)
    sym = sym.reshape(-1, 1)
    assert pmf.shape[0] == sym.shape[0]
    relevant_probabilities = torch.gather(pmf, dim=1, index=sym)
    bitrate = torch.sum(-torch.log2(relevant_probabilities.clamp(min=1e-3)))
    return bitrate
