import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
from hypercomp import models
from hypercomp import metrics
import wandb


def load_outer_model(artifact_id):
    run = wandb.init(project="MastersThesis")
    artifact = run.use_artifact(
        artifact_id, type='model')
    artifact_dir = artifact.download()
    inner_model = models.Conv1DModel(
        nChannels=p.CHANNELS, bpp_2=False)
    conv_model = models.LitAutoEncoder(inner_model, lr=p.LR)
    conv_model.load_from_checkpoint(
        artifact_dir+"/model.ckpt", model=inner_model)
    conv_model.train()
    conv_model.to(torch.device("cuda:"+str(p.GPU_ID)))
    conv_model.freeze()
    # for param in conv_model.autoencoder.encoder.parameters():
    #    param.requires_grad = False
    # conv_model.autoencoder.encoder.eval()
    return conv_model.autoencoder


if __name__ == "__main__":
    outer_model = load_outer_model(
        "niklas-sprengel/MastersThesis/model-12bfh33j:v0")

    model = models.LitAutoEncoder(models.CombinedModelWithHyperprior(
        nChannels=p.CHANNELS, innerChannels=26, H=128, W=128, outerModel=outer_model),
        lr=p.LR, loss=metrics.RateDistortionLoss(lmbda=p.RATE_DISTORTION_LDMBA), model_type=models.ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR)
    summary(model.autoencoder, input_size=(
        p.BATCH_SIZE, p.CHANNELS, 128, 128), device="cuda:"+str(p.GPU_ID))

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

    train_dataloader = data.dataLoader(
        train_dataset, batch_size=p.BATCH_SIZE)
    val_dataloader = data.dataLoader(
        val_dataset, batch_size=p.BATCH_SIZE)
    test_dataloader = data.dataLoader(
        test_dataset, batch_size=p.BATCH_SIZE)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
