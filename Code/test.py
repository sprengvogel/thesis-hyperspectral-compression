import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
from torch.utils.data import random_split
import math

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv1DModel(nChannels=369), lr=p.LR)
    summary(model.autoencoder, input_size=(1000, 1, 369))

    dataset = data.MatDataset(p.DATA_FOLDER, spacial_flatten=True)
    data_len = len(dataset)
    train_len = math.floor(data_len*0.5)
    val_len = math.floor(data_len*0.3)
    train_dataset, val_dataset, test_dataset = random_split(
        dataset, lengths=[train_len, val_len, data_len-train_len-val_len])
    train_dataloader = data.dataLoader(train_dataset)
    val_dataloader = data.dataLoader(val_dataset)
    test_dataloader = data.dataLoader(test_dataset)

    wandb_logger = WandbLogger(project="MastersThesis")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
