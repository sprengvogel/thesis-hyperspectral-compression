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
    inner_model = models.Conv1DModel(
        nChannels=p.CHANNELS, num_poolings=4)
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
        "niklas-sprengel/MastersThesis/model-3gm16mbp:v1")

    model = models.LitAutoEncoder(models.CombinedGeneralHyperprior(
        nChannels=p.CHANNELS, outerModel=outer_model,
        innerModel=models.GeneralHyperprior(main_autoencoder=models.ChengMain(in_channels=13),
                                            hyperprior_autoencoder=models.ChengHyperprior())),
                                  # loss=metrics.RateDistortionLoss(lmbda=p.RATE_DISTORTION_LDMBA),
                                  loss=metrics.MSELossWithBPPEstimation(),
                                  lr=p.LR,
                                  model_type=models.ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR)

    data.train_and_test(model, do_summary=True)
