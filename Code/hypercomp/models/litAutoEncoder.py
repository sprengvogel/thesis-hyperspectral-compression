import torch
import pytorch_lightning as pl
from .. import params as p
from ..metrics import psnr
import math


class LitAutoEncoder(pl.LightningModule):

    def __init__(self, model, lr) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = model
        self.loss = torch.nn.MSELoss()
        self.lr = lr

    def forward(self, x):
        return self.autoencoder.encoder(x)

    def training_step(self, batch, batch_idx):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x, x_hat)
        psnr_val = psnr(x, x_hat)
        self.log("train_loss", loss)
        self.log("psnr", psnr_val, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x, x_hat)
        psnr_val = psnr(x, x_hat)
        self.log(f"{prefix}_loss", loss)
        self.log(f"{prefix}_psnr", psnr_val, prog_bar=True)
        # Only log image once every epoch
        if batch_idx == 0:
            img1 = unflattenSpacialDimensionsIfNecessary(x[0])
            img1_hat = unflattenSpacialDimensionsIfNecessary(x_hat[0])
            # Log channel 100 for an image
            self.logger.log_image(key=f"{prefix}_sample", images=[
                img1[:, :, 100], img1_hat[:, :, 100]])

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)


"""
Takes a tensor of shape (H*W,C) and transforms it to (H,W,C). If not of this shape, returns original tensor
"""


def unflattenSpacialDimensionsIfNecessary(x: torch.Tensor) -> torch.Tensor:
    if x.dim() == 2:
        H = int(math.sqrt(x.shape[0]))
        return x.reshape((H, H, -1))
    return x
