import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv1DModel(nChannels=369), lr=p.LR)
    summary(model.autoencoder, input_size=(1000, 1, 369))

    dataset = data.MatDataset("hypercomp/data/mat-data/", spacial_flatten=True)
    dataloader = data.dataLoader(dataset)

    wandb_logger = WandbLogger(project="MastersThesis")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, dataloader)
