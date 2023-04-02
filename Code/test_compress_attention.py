import hypercomp.models as models
import hypercomp.data as data
import hypercomp.metrics as metrics
from hypercomp import params as p
import wandb
import torch
import numpy as np
from PIL import Image
import itertools


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


if __name__ == "__main__":
    run = wandb.init(project="MastersThesisTesting")
    artifact = run.use_artifact(
        "niklas-sprengel/MastersThesis/model-ebm7arkx:v0", type='model')
    artifact_dir = artifact.download()
    model = models.LitAutoEncoder(models.CombinedModelWithAttention(
        nChannels=p.CHANNELS, innerChannels=13),
        lr=p.LR, loss=metrics.RateDistortionLoss(lmbda=p.RATE_DISTORTION_LDMBA), model_type=models.ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR)
    model.load_from_checkpoint(
        artifact_dir+"/model.ckpt", model=model.autoencoder)
    model.freeze()

    updated = model.autoencoder.update()
    print(updated)

    test_dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
    test_dataloader = data.dataLoader(
        test_dataset, batch_size=1)

    iter = iter(test_dataloader)
    batch_1 = next(itertools.islice(iter, 35, None))
    print(batch_1.shape)
    in_img = Image.fromarray(convertVNIRImageToRGB(batch_1[0]))
    in_img.save("test_in.jpeg")
    compressed = model.autoencoder.compress(batch_1)
    strings = compressed["strings"]
    shape = compressed["shape"]
    with open("strings_test.txt", "wb") as f:
        f.write(strings[0][0])
        f.write(strings[1][0])
    print(strings)
    print(shape)
    output = model.autoencoder.decompress(strings, shape)
    print(output.shape)
    out_img = Image.fromarray(convertVNIRImageToRGB(output[0]))
    out_img.save("test_out.jpeg")
