import torch
import math
from torch.nn.functional import mse_loss
from torchmetrics import SpectralAngleMapper
import pytorch_msssim as mss
from math import degrees


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
    return degrees(sam(img_batch1, img_batch2))


class VAELoss(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target):
        x_hat, mean, log_variance = output
        # print(f"x_hat:{x_hat.shape}")
        # print(f"mean:{mean.shape}")
        # print(f"log_variance:{log_variance.shape}")
        # print(f"target:{target.shape}")
        recon_loss = self.mse(x_hat, target)
        mean = mean.squeeze()
        log_variance = log_variance.squeeze()
        KLD = torch.mean(-0.5 * torch.sum(1 + log_variance -
                                          mean.pow(2) - log_variance.exp(), dim=1), dim=0)
        print(recon_loss.shape)
        print(KLD.shape)
        return recon_loss+KLD, recon_loss, KLD


class DualMSELoss(torch.nn.Module):
    """Custom mse loss that calculates the mse loss for an outer and an inner compression network as lambda * mse(outer_net) + (1-lambda) * mse(inner_net)."""

    def __init__(self, lmbda=0.5):
        super().__init__()
        self.mse = torch.nn.MSELoss()
        self.lmbda = lmbda

    def forward(self, output, target):
        x_hat_outer, x_hat_inner, x_inner = output

        mse_loss_outer = self.mse(x_hat_outer, target)
        mse_loss_inner = self.mse(x_hat_inner, x_inner)
        dual_mse_loss = self.lmbda * mse_loss_outer + \
            (1-self.lmbda) * mse_loss_inner

        return dual_mse_loss, mse_loss_inner, mse_loss_outer


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
        elif len(output) == 5:
            x_hat, _, _, y_likelihoods, z_likelihoods = output
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


class MSELossWithBPPEstimation(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.mse = torch.nn.MSELoss()

    def forward(self, output, target):
        x_hat, x_hat_inner, latent_image, y_likelihoods, z_likelihoods = output
        likelihoods = [y_likelihoods, z_likelihoods]
        N, _, H, W = target.size()
        num_pixels = N * H * W

        bpp_loss = sum(
            (torch.log(likelihood).sum() / (-math.log(2) * num_pixels))
            for likelihood in likelihoods
        )

        mse_loss = self.mse(x_hat, target)
        return mse_loss, mse_loss, bpp_loss
