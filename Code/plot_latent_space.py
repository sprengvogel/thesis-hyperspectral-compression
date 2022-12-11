import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from hypercomp.models.utils import unflatten_and_split_apart_batches
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import random_split
import math
import numpy as np
from hypercomp import metrics
import wandb
import matplotlib.pyplot as plt


@torch.no_grad()
def encodeWithConvModel(x: torch.Tensor, conv_model: models.LitAutoEncoder, scale=False):
    x = x.to(torch.device("cuda:3"))
    out = unflatten_and_split_apart_batches(
        conv_model(x)).squeeze()
    if scale:
        # Scale tensor to [0,1]
        out -= out.min()
        out /= out.max()
    return out.detach().cpu().numpy()


def convertVNIRImageToRGB(hyperspectral_image: np.ndarray, conv_output=False):
    """
    Extracts a red, green and blue channel from a picture from the VNIR sensor (369 channels between 400-1000nm).
    Because it is needed in channel-last format for wandb logger, this returns (H,W,C).
    """
    r_id, g_id, b_id = 154, 74, 34
    if conv_output:
        r_id //= 3
        g_id //= 3
        b_id //= 3
    red_channel = hyperspectral_image[r_id, :, :]
    green_channel = hyperspectral_image[g_id, :, :]
    blue_channel = hyperspectral_image[b_id, :, :]
    return np.uint8((np.stack([red_channel, green_channel, blue_channel], axis=-1)*255))


if __name__ == "__main__":
    run = wandb.init(project="MastersThesis", job_type="Latent_Space_Plot")
    artifact = run.use_artifact(
        'niklas-sprengel/MastersThesis/model-1x92mdk2:v0', type='model')
    artifact_dir = artifact.download()
    conv_model = models.LitAutoEncoder(
        models.Conv1DModel(nChannels=369), lr=p.LR)
    conv_model.load_from_checkpoint(artifact_dir+"/model.ckpt")
    conv_model.eval()
    conv_model.freeze()
    conv_model.to(torch.device("cuda:3"))

    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test", transform=None)

    test_dataloader = data.dataLoader(
        test_dataset, batch_size=1)

    pixel_x, pixel_y = 40, 40
    input_values = []
    output_values = []
    input_imgs = []
    output_imgs = []
    for img in test_dataloader:
        input_values.append(img.squeeze().detach().cpu().numpy()[
                            :, pixel_x, pixel_y])
        input_imgs.append(convertVNIRImageToRGB(
            img.squeeze().detach().cpu().numpy()))
        output = encodeWithConvModel(img, conv_model=conv_model, scale=True)
        output_imgs.append(convertVNIRImageToRGB(output, conv_output=True))
        output_values.append(output.squeeze()[:, pixel_x, pixel_y])
    input_values = np.array(input_values)
    input_values = np.mean(input_values, axis=0)
    output_values = np.array(output_values)
    output_values = np.mean(output_values, axis=0)
    x_axis_input = range(input_values.shape[0])
    fig = plt.figure()
    plt.subplot(211)
    plt.bar(x_axis_input, input_values)
    plt.title("Spectral distribution of original images")
    x_axis_output = range(output_values.shape[0])
    plt.subplot(212)
    plt.bar(x_axis_output, output_values)
    plt.title("Spectral distribution of latent space.")
    wandb.log({"chart": fig})
    for in_img, out_img in zip(input_imgs, output_imgs):
        wandb.log(
            {"input left , output right": [wandb.Image(in_img), wandb.Image(out_img)]})
