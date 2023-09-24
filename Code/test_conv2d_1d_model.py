import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p
from hypercomp.models.utils import unflatten_and_split_apart_batches, flatten_spacial_dims
import torch
import numpy as np
import wandb
from hypercomp import metrics as metrics


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
    run = wandb.init(project="MastersThesis",
                     job_type="Test_Conv1d+2d")
    artifact = run.use_artifact(
        'niklas-sprengel/MastersThesis/model-1x92mdk2:v0', type='model')
    artifact_dir = artifact.download()
    conv_model = models.LitAutoEncoder(
        models.Conv1DModel(nChannels=369), lr=p.LR)
    conv_model.load_from_checkpoint(artifact_dir+"/model.ckpt")
    conv_model.eval()
    conv_model.freeze()
    conv_model.to(torch.device("cuda:"+str(p.GPU_ID)))

    artifact2 = run.use_artifact(
        'niklas-sprengel/MastersThesis/model-2pz9sk76:v21', type='model')
    artifact_dir2 = artifact2.download()
    model = models.LitAutoEncoder(models.Conv2DModel(
        nChannels=93, H=96, W=96), lr=p.LR)
    model.load_from_checkpoint(
        artifact_dir2+"/model.ckpt", model=model.autoencoder)
    model.eval()
    model.freeze()
    model.to(torch.device("cuda:"+str(p.GPU_ID)))

    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test", transform=None)

    test_dataloader = data.dataLoader(
        test_dataset, batch_size=1)

    input_imgs, output_imgs = [], []
    psnrs = []
    for img in test_dataloader:
        input_imgs.append(convertVNIRImageToRGB(
            img.squeeze().detach().cpu().numpy()))

        output = unflatten_and_split_apart_batches(
            conv_model.autoencoder.encoder(img.to(torch.device("cuda:"+str(p.GPU_ID)))))
        output = model.autoencoder(output)
        output = conv_model.autoencoder.decoder(flatten_spacial_dims(output))
        print(output)

        psnrs.append(metrics.psnr(img.detach().cpu(), output.detach().cpu()))
        output_imgs.append(convertVNIRImageToRGB(
            output.squeeze().cpu().detach().numpy(), conv_output=False))

    wandb.log({"psnr": psnrs})
    for in_img, out_img in zip(input_imgs, output_imgs):
        wandb.log(
            {"input left , output right": [wandb.Image(in_img), wandb.Image(out_img)]})
