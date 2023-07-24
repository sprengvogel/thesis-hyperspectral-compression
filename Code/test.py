from hypercomp import data
import torch
from pytorch_lightning.loggers import WandbLogger
from hypercomp import params as p
from hypercomp import models
from hypercomp import metrics
import wandb
import pytorch_lightning as pl

dataset = data.HySpecNet11k(
    root_dir=p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")
dataloader = data.dataLoader(dataset, shuffle=False)
run = wandb.init(project="MastersThesis")
artifact = run.use_artifact(
    "niklas-sprengel/MastersThesis/model-214yd7yf:v0", type='model')
artifact_dir = artifact.download()
onedmodel = models.Fast1DConvModel(
    nChannels=p.CHANNELS, bottleneck_size=3, H=128, W=128)
model = models.FastCombinedModel(
    nChannels=p.CHANNELS, bottleneck_size=3, H=128, W=128, kernel_size=3, outerModel=onedmodel)
loaded_model = models.LitAutoEncoder.load_from_checkpoint(
    artifact_dir+"/model.ckpt", model=model, loss=metrics.DualMSELoss(lmbda=p.DUAL_MSE_LOSS_LMBDA))
wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
accelerator = "gpu" if torch.cuda.is_available() else "cpu"
trainer = pl.Trainer(accelerator=accelerator, logger=wandb_logger,
                     log_every_n_steps=50, devices=[p.GPU_ID])
trainer.test(loaded_model, dataloaders=dataloader)
