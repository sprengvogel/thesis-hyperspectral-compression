import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint
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
    inner_model = models.Fast1DConvModel(
        nChannels=p.CHANNELS, bottleneck_size=13)
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
        nChannels=p.CHANNELS, innerChannels=13, H=128, W=128, outerModel=outer_model),
        lr=p.LR, loss=metrics.RateDistortionLoss(lmbda=p.RATE_DISTORTION_LDMBA), model_type=models.ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR)
    data.train_and_test(model)