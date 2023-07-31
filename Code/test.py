from hypercomp import data
import torch
from pytorch_lightning.loggers import WandbLogger
from hypercomp import params as p
from hypercomp import models
from hypercomp import metrics
import wandb
import pytorch_lightning as pl


def createFastCombinedModel(bottleneck_size: int):
    onedmodel = models.Fast1DConvModel(
        nChannels=p.CHANNELS, bottleneck_size=bottleneck_size, H=128, W=128)
    return models.FastCombinedModel(
        nChannels=p.CHANNELS, bottleneck_size=bottleneck_size, H=128, W=128, kernel_size=p.KERNEL_SIZE_CONV2D, outerModel=onedmodel)


def createCombinedModel(num_poolings: int, inner_channels: int):
    onedmodel = models.Conv1DModel(
        nChannels=p.CHANNELS, num_poolings=num_poolings)
    return models.CombinedModel(
        nChannels=p.CHANNELS, innerChannels=inner_channels, H=128, W=128, outerModel=onedmodel)


dataset = data.HySpecNet11k(
    root_dir=p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
dataloader = data.dataLoader(dataset, shuffle=False)
run = wandb.init(project="MastersThesis")
artifact = run.use_artifact(
    "niklas-sprengel/MastersThesis/model-10amsm19:v0", type='model')
artifact_dir = artifact.download()
model = createFastCombinedModel(6)
# model = createCombinedModel(4, 13)
loaded_model = models.LitAutoEncoder.load_from_checkpoint(
    artifact_dir+"/model.ckpt", model=model, loss=metrics.DualMSELoss(lmbda=p.DUAL_MSE_LOSS_LMBDA), log_all_imgs=True)
wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(accelerator=accelerator, logger=wandb_logger,
                     log_every_n_steps=50, devices=[p.GPU_ID])
trainer.test(loaded_model, dataloaders=dataloader)
