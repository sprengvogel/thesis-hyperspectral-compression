import torch
import math
from torch.nn.functional import mse_loss

"""
Computes the peak signal to noise ratio (PSNR).
max_val is the maximum possible pixel value, in hyperspectral datasets often 1.
"""


def psnr(img1, img2, max_val=1):
    mse = mse_loss(img1, img2, reduction='none')
    # Add 1e-8 in log for numerical stability
    psnr = 20*math.log10(max_val)-10*torch.log10(mse + 1e-8)
    # Average over batches
    return torch.mean(psnr).item()
