import torch
import numpy as np
import pytorch_lightning as pl
from ..models import LitAutoEncoder
from .hySpecNet11k import HySpecNet11k
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
from torch.utils.data import Dataset, DataLoader
from .. import params as p


def dataLoader(dataset: Dataset, batch_size = p.BATCH_SIZE, shuffle = p.SHUFFLE_DATA_LOADER) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=p.NUM_WORKERS, pin_memory=True, drop_last=True)

def train_and_test(model: LitAutoEncoder, batch_size=p.BATCH_SIZE, do_summary = True):
    pl.seed_everything(0)
    if do_summary:
        summary(model.autoencoder, input_size=(8,
            p.CHANNELS, 128, 128), device="cuda:"+str(p.GPU_ID))

    train_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="train")
    val_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="val")
    test_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")

    train_dataloader = dataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = dataLoader(val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = dataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    checkpoint_callback = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor="val_metrics/psnr", mode="max")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(gradient_clip_val=1.0,
                         accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID], callbacks=[checkpoint_callback])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)