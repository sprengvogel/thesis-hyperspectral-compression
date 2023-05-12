import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from hypercomp import metrics
from torch.utils.data import DataLoader
from torch.nn.functional import mse_loss
import numpy as np
import wandb
import torch
from tqdm import tqdm
import math


def convertVNIRImageToRGB(hyperspectral_image: torch.Tensor):
    """
    Extracts a red, green and blue channel from a picture from the VNIR sensor (369 channels between 400-1000nm).
    Because it is needed in channel-last format for wandb logger, this returns (H,W,C).
    """
    if hyperspectral_image.shape[0] == 369:
        red_channel = hyperspectral_image[154, :, :]
        green_channel = hyperspectral_image[74, :, :]
        blue_channel = hyperspectral_image[34, :, :]
    elif hyperspectral_image.shape[0] == 202:
        red_channel = hyperspectral_image[44, :, :]
        green_channel = hyperspectral_image[29, :, :]
        blue_channel = hyperspectral_image[11, :, :]
    else:
        raise ValueError("Not a known number of channels.")
    return np.uint8((torch.stack([red_channel, green_channel, blue_channel], dim=-1)*255).cpu().detach().numpy())


def compute_split_points(big_channels, small_channels):
    """
    Computes the split points used for interpolation and restoration
    """
    # We need one extra split point, because both start and end are included
    return np.linspace(start=0, stop=big_channels-1, num=small_channels+1, endpoint=True, dtype=int)


def interpolate_image(img_batch, n_output_channels):
    """
    Interpolate image channels in a naive manner
    Input: Img batch with dimensions (B, C, H, W)
    Output: Interpolated img batch with dimensions (B, n_output_channels, H, W)
    """
    in_channels = img_batch.shape[1]
    split_points = compute_split_points(
        big_channels=in_channels, small_channels=n_output_channels)
    interpolated_values = [np.average(img_batch[:, split_points[i]:split_points[i+1], :, :], axis=1)
                           for i in range(len(split_points)-1)]
    return np.stack(interpolated_values, axis=1)


def decode_interpolated_image(img_batch, n_output_channels=p.CHANNELS):
    """
    'Decodes' interpolated image by setting the value of the interpolated channel for all channels it is interpolated from
    """
    in_channels = img_batch.shape[1]
    result = np.empty(
        (img_batch.shape[0], n_output_channels, img_batch.shape[2], img_batch.shape[3]))
    split_points = compute_split_points(
        big_channels=n_output_channels, small_channels=in_channels)
    for i in range(len(split_points)-1):
        split_point = split_points[i]
        next_point = split_points[i+1]
        result[:, split_point:next_point, :, :] = np.expand_dims(
            img_batch[:, i, :, :], 1)
    return result


if __name__ == "__main__":
    batch_size = 128
    latent_channels = 56
    run = wandb.init(project="MastersThesis", tags=[
                     "baseline", "interpolation"], name=f"InterpolationLatent{latent_channels}")
    dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=False, drop_last=False)
    mse = []
    psnr = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        latent = interpolate_image(
            batch.numpy(), n_output_channels=latent_channels)
        rec = decode_interpolated_image(latent)
        rec = torch.from_numpy(rec)
        assert latent.shape[1:] == (latent_channels, 128, 128)
        assert rec.shape == batch.shape
        mse.append(mse_loss(batch, rec))
        psnr.append(metrics.psnr(batch, rec))
        if batch_idx < 5:
            run.log(
                {f"test_images/sample{batch_idx}": [wandb.Image(convertVNIRImageToRGB(batch[0])), wandb.Image(convertVNIRImageToRGB(rec[0]))]})
            run.log(
                {f"test_images/latent{batch_idx}": [wandb.Image(latent[0, i, :, :]) for i in range(min(latent_channels, 13))]})
    run.log({"test_loss/loss": np.mean(mse), "test_loss/mse": np.mean(mse),
            "test_metrics/psnr": np.mean(psnr), "test_metrics/bpppc": 32*latent_channels/p.CHANNELS})
