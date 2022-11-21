import torch
import pytorch_lightning as pl
from .. import params as p


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
        self.log("train_loss", loss)
        return loss

    def validation_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        self._shared_eval(batch, batch_idx, "test")

    def _shared_eval(self, batch, batch_idx, prefix):
        x = batch
        x_hat = self.autoencoder(x)
        loss = self.loss(x, x_hat)
        self.log(f"{prefix}_loss", loss)

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.lr)
