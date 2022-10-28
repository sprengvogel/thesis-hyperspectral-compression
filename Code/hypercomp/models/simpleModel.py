import torch
import pytorch_lightning as pl


class LitSimpleModel(pl.LightningModule):

    def __init__(self) -> None:
        super().__init__()
        self.save_hyperparameters()
        self.autoencoder = SimpleModel()
        self.loss = torch.nn.MSELoss()

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
        return torch.optim.Adam(self.parameters(), lr=0.02)


class SimpleModel(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()

        self.encoder = SimpleEncoder()
        self.decoder = SimpleDecoder()

    def forward(self, x):
        return self.decoder(self.encoder(x))


class SimpleEncoder(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.Conv2d(369, 123, kernel_size=1)

    def forward(self, x):
        return self.conv1(x)


class SimpleDecoder(torch.nn.Module):

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = torch.nn.ConvTranspose2d(123, 369, kernel_size=1)

    def forward(self, x):
        return self.conv1(x)
