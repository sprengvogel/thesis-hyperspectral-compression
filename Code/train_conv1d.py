import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import random_split
import math
import numpy as np

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv1DModel(nChannels=369), lr=p.LR)
    summary(model.autoencoder, input_size=(p.BATCH_SIZE,369,96,96))

    train_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="train")
    val_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="val")
    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test")

    train_dataloader = data.dataLoader(train_dataset)
    val_dataloader = data.dataLoader(val_dataset)
    test_dataloader = data.dataLoader(test_dataset)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[3])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
