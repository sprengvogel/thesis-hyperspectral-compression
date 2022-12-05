import torch
import math
from torch.nn.functional import mse_loss
from torchmetrics import SpectralAngleMapper
import pytorch_msssim as mss

"""
Computes the peak signal to noise ratio (PSNR).
max_val is the maximum possible pixel value, in hyperspectral datasets often 1.
"""


def psnr(img1, img2, max_val=1):
    # Average mse over batches
    mse = mse_loss(img1, img2, reduction='mean')
    # Add 1e-8 in log for numerical stability
    psnr = 20*math.log10(max_val)-10*torch.log10(mse + 1e-8)
    return psnr.item()


def ssim(img1, img2):
    return mss.ssim(X=img1, Y=img2, data_range=1.0, nonnegative_ssim=True)


def spectral_angle(img1, img2):
    sam = SpectralAngleMapper(reduction="elementwise_mean")
    return sam(img1, img2)
