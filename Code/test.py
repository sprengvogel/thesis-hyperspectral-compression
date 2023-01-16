from hypercomp import data
import torch
from pytorch_lightning.loggers import WandbLogger
import numpy as np


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
        red_channel = hyperspectral_image[100, :, :]
        green_channel = hyperspectral_image[50, :, :]
        blue_channel = hyperspectral_image[20, :, :]
    else:
        raise ValueError("Not a known number of channels.")
    return np.uint8((torch.stack([red_channel, green_channel, blue_channel], dim=-1)*255).cpu().numpy())


dataset = data.HySpecNet11k(
    root_dir="/media/storagecube/data/datasets/enmap/dataset", mode="easy", split="train")
dataloader = data.dataLoader(dataset)
logger = WandbLogger("test")
for i in dataloader:
    im = convertVNIRImageToRGB(i[0])
    logger.log_image(key="test", images=[im])
