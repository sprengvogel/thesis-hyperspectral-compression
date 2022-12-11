import torch
import math
from torch.nn.functional import mse_loss
from torchmetrics import SpectralAngleMapper
import pytorch_msssim as mss


def psnr(img_batch1, img_batch2, max_val=1):
    """
    Computes the peak signal to noise ratio (PSNR).
    max_val is the maximum possible pixel value, in hyperspectral datasets often 1.
    """
    # Average mse over batches
    mse = mse_loss(img_batch1, img_batch2, reduction='mean')
    # Add 1e-8 in log for numerical stability
    psnr = 20*math.log10(max_val)-10*torch.log10(mse + 1e-8)
    return psnr.item()


def ssim(img_batch1, img_batch2):
    return mss.ssim(X=img_batch1, Y=img_batch2, data_range=1.0, nonnegative_ssim=True)


def spectral_angle(img_batch1, img_batch2):
    sam = SpectralAngleMapper(reduction="elementwise_mean")
    return sam(img_batch1, img_batch2)


class RateDistortionLoss(torch.nn.Module):
    """Custom rate distortion loss with a Lagrangian parameter."""

    def __init__(self, lmbda=1e-2):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        if len(output) == 3:
            x_hat, y_likelihoods, z_likelihoods = output
            likelihoods = [y_likelihoods, z_likelihoods]
        else:
            x_hat, likelihoods = output
        N, _, H, W = target.size()
        num_pixels = N * H * W

        bpp_loss = sum(
            (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
            for likelihood in likelihoods
        )
        mse_loss = self.mse(x_hat, target)
        rate_distortion = self.lmbda * 255**2 * mse_loss + bpp_loss

        return rate_distortion, mse_loss, bpp_loss
