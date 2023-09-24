import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from hypercomp import metrics
from hypercomp import models

if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    model = models.LitAutoEncoder(models.ScaleHyperprior(
        channel_number=369, N=128, M=192), lr=p.LR_HYPERPRIOR, loss=metrics.RateDistortionLoss(lmbda=1), model_type=models.ModelType.HYPERPRIOR)
    summary(model.autoencoder, input_size=(p.BATCH_SIZE, 369, 96, 96))

    train_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="train")
    val_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="val")
    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test")

    train_dataloader = data.dataLoader(
        train_dataset, batch_size=p.BATCH_SIZE_HYPERPRIOR)
    val_dataloader = data.dataLoader(
        val_dataset, batch_size=p.BATCH_SIZE_HYPERPRIOR)
    test_dataloader = data.dataLoader(
        test_dataset, batch_size=p.BATCH_SIZE_HYPERPRIOR)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
