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
import argparse
import scipy.interpolate as ip


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
    return np.linspace(start=0, stop=big_channels-1, num=small_channels, endpoint=True, dtype=int)


def compress_image_with_averaging(img_batch, n_output_channels):
    """
    Compress image by averaging values between split points
    Input: Img batch with dimensions (B, C, H, W)
    Output: Compressed img batch with dimensions (B, n_output_channels, H, W)
    """
    in_channels = img_batch.shape[1]
    # We need one extra split point, because both start and end are included
    split_points = compute_split_points(
        big_channels=in_channels, small_channels=n_output_channels+1)
    interpolated_values = [np.average(img_batch[:, split_points[i]:split_points[i+1], :, :], axis=1)
                           for i in range(len(split_points)-1)]
    return np.stack(interpolated_values, axis=1)


def decode_averaged_image(img_batch, n_output_channels=p.CHANNELS):
    """
    'Decodes' interpolated image by setting the value of the interpolated channel for all channels it is interpolated from
    """
    in_channels = img_batch.shape[1]
    result = np.empty(
        (img_batch.shape[0], n_output_channels, img_batch.shape[2], img_batch.shape[3]))
    # We need one extra split point, because both start and end are included
    split_points = compute_split_points(
        big_channels=n_output_channels, small_channels=in_channels+1)
    for i in range(len(split_points)-1):
        split_point = split_points[i]
        next_point = split_points[i+1]
        result[:, split_point:next_point, :, :] = np.expand_dims(
            img_batch[:, i, :, :], 1)
    return result


def encode_general_interpolation(img_batch, n_output_channels):
    """
    Encodes by throwing away all points except for @n_output_channels number of points, chosen equidistant
    Input: Img batch with dimensions (B, C, H, W)
    Output: Interpolated img batch with dimensions (B, n_output_channels, H, W)
    """
    in_channels = img_batch.shape[1]
    split_points = compute_split_points(
        big_channels=in_channels, small_channels=n_output_channels)
    return img_batch[:, split_points, :, :]


def decode_linear_interpolation(img_batch, n_output_channels=p.CHANNELS):
    """
    Performs linear interpolation on the image batch
    """
    in_channels = img_batch.shape[1]
    result = np.empty(
        (img_batch.shape[0], n_output_channels, img_batch.shape[2], img_batch.shape[3]))
    split_points = compute_split_points(
        big_channels=n_output_channels, small_channels=in_channels)
    for i in range(len(split_points)-1):
        split_point = split_points[i]
        next_point = split_points[i+1]
        delta = img_batch[:, i+1, :, :] - img_batch[:, i, :, :]
        gradient = delta / (next_point-split_point)
        for j in range(next_point-split_point):
            result[:, split_point+j, :, :] = img_batch[:, i, :, :] + j*gradient
    return result


def decode_general_interpolation(img_batch, kind: str, n_output_channels=p.CHANNELS):
    """
    Performs different kinds of interpolation on the image batch
    """
    in_channels = img_batch.shape[1]
    result = np.empty(
        (img_batch.shape[0], n_output_channels, img_batch.shape[2], img_batch.shape[3]))
    split_points = compute_split_points(
        big_channels=n_output_channels, small_channels=in_channels)
    for b in range(img_batch.shape[0]):
        for x in range(img_batch.shape[2]):
            for y in range(img_batch.shape[3]):
                f = ip.interp1d(
                    split_points, img_batch[b, :, x, y], kind=kind)
                result[b, :, x, y] = f(np.arange(n_output_channels))
    return result


def main(batch_size: int, latent_channels: int, interpolation_mode: str):
    run = wandb.init(project="MastersThesis", tags=[
                     "baseline", "interpolation"], name=f"{interpolation_mode.capitalize()}InterpolationLatent{latent_channels}")
    dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
    dataloader = DataLoader(dataset, batch_size=batch_size,
                            shuffle=False, pin_memory=False, drop_last=False)
    mse = []
    psnr = []
    for batch_idx, batch in enumerate(tqdm(dataloader)):
        if interpolation_mode == "average":
            latent = compress_image_with_averaging(
                batch.numpy(), n_output_channels=latent_channels)
            rec = decode_averaged_image(latent)
        elif interpolation_mode == "linear":
            latent = encode_general_interpolation(
                batch.numpy(), n_output_channels=latent_channels)
            rec = decode_linear_interpolation(latent)
        elif interpolation_mode == "linear2":
            latent = encode_general_interpolation(
                batch.numpy(), n_output_channels=latent_channels)
            rec = decode_general_interpolation(latent, kind="linear")
        elif interpolation_mode == "cubic":
            latent = encode_general_interpolation(
                batch.numpy(), n_output_channels=latent_channels)
            rec = decode_general_interpolation(latent, kind="cubic")
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


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description='Evaluate interpolation methods on test set')
    parser.add_argument(
        "-b",
        "--batchsize",
        type=int,
        default=128,
        help="Batch size",
    )
    parser.add_argument(
        "-c",
        '--channels',
        default=13,
        type=int,
        help='Number of desired latent channels')
    parser.add_argument(
        "-m",
        "--mode",
        default="linear",
        type=str,
        choices=["linear", "average", "linear2", "cubic"],
        help="Interpolation algorithm (default: %(default)s)",
    )

    parse_args = parser.parse_args()
    main(parse_args.batchsize, parse_args.channels, parse_args.mode)
