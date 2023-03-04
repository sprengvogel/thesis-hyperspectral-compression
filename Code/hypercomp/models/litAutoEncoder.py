import torch
import pytorch_lightning as pl
from torchvision.utils import make_grid
import wandb
import numpy as np
from .. import params as p
from ..metrics import psnr, ssim, spectral_angle
from .modelType import ModelType


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, model: torch.nn.Module, lr: float, loss=torch.nn.MSELoss(), model_type: ModelType = ModelType.OTHER, weight_decay=p.WEIGHT_DECAY, dual_mse_loss_lambda=p.DUAL_MSE_LOSS_LMBDA, rate_distortion_ldmba=p.RATE_DISTORTION_LDMBA) -> None:
        super().__init__()
        self.save_hyperparameters(ignore=['model','loss'])
        self.autoencoder = model
        self.loss = loss
        self.lr = lr
        self.model_type = model_type

    def forward(self, x):
        return self.autoencoder.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x_hat, x)
        latent_image = None
        x_hat_inner = None
        if self.model_type == ModelType.HYPERPRIOR:
            loss, mse, bpp = loss
            x_hat = x_hat[0]
            self.log("train_loss/bpp", bpp, prog_bar=True)
            self.log("train_loss/mse", mse, prog_bar=True)
        elif self.model_type == ModelType.CONV1D_AND_2D:
            loss, inner_loss, outer_loss = loss
            x_hat = x_hat[0]
            latent_image = x_hat[2]
            x_hat_inner = x_hat[1]
            self.log("train_loss/dual_mse", loss, prog_bar=True)
            self.log("train_loss/inner_mse", inner_loss, prog_bar=True)
            self.log("train_loss/outer_mse", outer_loss, prog_bar=True)
        elif self.model_type == ModelType.VAE:
            loss, mse, kld = loss
            self.log("train_loss/mse", mse, prog_bar=True)
            self.log("train_loss/kld", kld, prog_bar=True)
            x_hat = x_hat[0]
        elif self.model_type == ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR:
            loss, mse, bpp = loss
            x_hat = x_hat[0]
            latent_image = x_hat[2]
            x_hat_inner = x_hat[1]
            self.log("train_loss/bpp", bpp, prog_bar=True)
            self.log("train_loss/mse", mse, prog_bar=False)
        psnr_val = psnr(x_hat, x)
        #ssim_val = ssim(x_hat, x)
        spectral_angle_val = spectral_angle(x_hat, x)
        self.log("train_loss/loss", loss)
        self.log("train_metrics/psnr", psnr_val, prog_bar=True)
        #self.log("train_metrics/ssim", ssim_val, prog_bar=True)
        self.log("train_metrics/spectral angle",
                 spectral_angle_val, prog_bar=False)
        # Only log image once every epoch
        if batch_idx > 50 and batch_idx < 55:
            img = x[0]
            img_hat = x_hat[0]
            # Log rgb version of an image
            self.logger.log_image(key=f"train_images/sample_{batch_idx}", images=[
                convertVNIRImageToRGB(img), convertVNIRImageToRGB(img_hat)])
            if x_hat_inner is not None:
                inner_img = latent_image[0]
                inner_img_hat = x_hat_inner[0]
                self.logger.log_image(key=f"train_latentimages/sample_{batch_idx}", images=[
                    convertVNIRImageToRGB(inner_img), convertVNIRImageToRGB(inner_img_hat)])
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x_hat, x)
        latent_image = None
        x_hat_inner = None
        if self.model_type == ModelType.HYPERPRIOR:
            loss, mse, bpp = loss
            x_hat = x_hat[0]
            self.log(f"{prefix}_loss/bpp", bpp, prog_bar=False)
            self.log(f"{prefix}_loss/mse", mse, prog_bar=False)
        elif self.model_type == ModelType.CONV1D_AND_2D:
            loss, inner_loss, outer_loss = loss
            x_hat = x_hat[0]
            latent_image = x_hat[2]
            x_hat_inner = x_hat[1]
            self.log(f"{prefix}_loss/dual_mse", loss, prog_bar=True)
            self.log(f"{prefix}_loss/inner_mse", inner_loss, prog_bar=True)
            self.log(f"{prefix}_loss/outer_mse", outer_loss, prog_bar=True)
        elif self.model_type == ModelType.VAE:
            loss, mse, kld = loss
            self.log(f"{prefix}_loss/mse", mse, prog_bar=True)
            self.log(f"{prefix}_loss/kld", kld, prog_bar=True)
            x_hat = x_hat[0]
        elif self.model_type == ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR:
            loss, mse_loss, bpp = loss
            x_hat = x_hat[0]
            latent_image = x_hat[2]
            x_hat_inner = x_hat[1]
            self.log(f"{prefix}_loss/bpp", bpp, prog_bar=True)
            self.log(f"{prefix}_loss/mse", mse_loss, prog_bar=False)
        psnr_val = psnr(x_hat, x)
        #ssim_val = ssim(x_hat, x)
        spectral_angle_val = spectral_angle(x_hat, x)
        self.log(f"{prefix}_loss/loss", loss)
        self.log(f"{prefix}_metrics/psnr", psnr_val, prog_bar=False)
        #self.log(f"{prefix}_metrics/ssim", ssim_val, prog_bar=False)
        self.log(f"{prefix}_metrics/spectral angle",
                 spectral_angle_val, prog_bar=False)
        # Only log image once every epoch
        if batch_idx > 50 and batch_idx < 55:
            img = x[0]
            img_hat = x_hat[0]
            # Log rgb version of an image
            self.logger.log_image(key=f"{prefix}_images/sample_{batch_idx}", images=[
                convertVNIRImageToRGB(img), convertVNIRImageToRGB(img_hat)])
            if x_hat_inner is not None:
                inner_img = latent_image[0]
                inner_img_hat = x_hat_inner[0]
                self.logger.log_image(key=f"{prefix}_latentimages/sample_{batch_idx}", images=[
                    convertVNIRImageToRGB(inner_img), convertVNIRImageToRGB(inner_img_hat)])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=p.WEIGHT_DECAY)


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
