import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
import wandb
import numpy as np
from .. import params as p
from ..metrics import psnr, ssim, spectral_angle


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, lr: float, loss=torch.nn.MSELoss(), hyperprior: bool = False) -> None:
        super().__init__()
        self.save_hyperparameters("lr")
        self.autoencoder = model
        self.loss = loss
        self.lr = lr
        self.hyperprior = hyperprior

    def forward(self, x):
        return self.autoencoder.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x_hat, x)
        if self.hyperprior:
            loss, mse, bpp = loss
            x_hat = x_hat[0]
            self.log("train_loss/bpp", bpp, prog_bar=True)
            self.log("train_loss/mse", mse, prog_bar=True)
        psnr_val = psnr(x_hat, x)
        ssim_val = ssim(x_hat, x)
        spectral_angle_val = spectral_angle(x_hat, x)
        self.log("train_loss/loss", loss)
        self.log("train_metrics/psnr", psnr_val, prog_bar=True)
        self.log("train_metrics/ssim", ssim_val, prog_bar=True)
        self.log("train_metrics/spectral angle",
                 spectral_angle_val, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x_hat, x)
        if self.hyperprior:
            loss, mse, bpp = loss
            x_hat = x_hat[0]
            self.log(f"{prefix}_loss/bpp", bpp, prog_bar=False)
            self.log(f"{prefix}_loss/mse", mse, prog_bar=False)
        psnr_val = psnr(x_hat, x)
        ssim_val = ssim(x_hat, x)
        spectral_angle_val = spectral_angle(x_hat, x)
        self.log(f"{prefix}_loss/loss", loss)
        self.log(f"{prefix}_metrics/psnr", psnr_val, prog_bar=False)
        self.log(f"{prefix}_metrics/ssim", ssim_val, prog_bar=False)
        self.log(f"{prefix}_metrics/spectral angle",
                 spectral_angle_val, prog_bar=False)
        # Only log image once every epoch
        if batch_idx < 4:
            img = x[0]
            img_hat = x_hat[0]
            # Log rgb version of an image
            self.logger.log_image(key=f"{prefix}_images/sample_{batch_idx}", images=[
                convertVNIRImageToRGB(img), convertVNIRImageToRGB(img_hat)])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


def convertVNIRImageToRGB(hyperspectral_image: torch.Tensor):
    """
    Extracts a red, green and blue channel from a picture from the VNIR sensor (369 channels between 400-1000nm).
    Because it is needed in channel-last format for wandb logger, this returns (H,W,C).
    """
    red_channel = hyperspectral_image[154//3, :, :]
    green_channel = hyperspectral_image[74//3, :, :]
    blue_channel = hyperspectral_image[34//3, :, :]
    return np.uint8((torch.stack([red_channel, green_channel, blue_channel], dim=-1)*255).cpu().numpy())
