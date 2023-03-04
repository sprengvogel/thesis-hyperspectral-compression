import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
import torch
from torch.utils.data import random_split
import math
import numpy as np

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv1DModel(
        nChannels=p.CHANNELS, bpp_2=True), lr=p.LR)
    summary(model.autoencoder, input_size=(p.BATCH_SIZE,
            p.CHANNELS, 128, 128), device="cuda:"+str(p.GPU_ID))

    """train_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="train")
    val_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="val")
    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test")"""

    train_dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="train")
    val_dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="val")
    test_dataset = data.HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")

    train_dataloader = data.dataLoader(train_dataset)
    val_dataloader = data.dataLoader(val_dataset)
    test_dataloader = data.dataLoader(test_dataset)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    checkpoint_callback = ModelCheckpoint(
        save_last=True, save_top_k=5, monitor="val_metrics/psnr", mode="max")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID], callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
