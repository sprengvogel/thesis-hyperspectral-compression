import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p
import torch
from hypercomp import models
from hypercomp import metrics
import wandb


def load_outer_model(artifact_id):
    run = wandb.init(project="MastersThesis")
    artifact = run.use_artifact(
        artifact_id, type='model')
    artifact_dir = artifact.download()
    inner_model = models.Fast1DConvModel(
        nChannels=p.CHANNELS, bottleneck_size=13, H=128, W=128)
    conv_model = models.LitAutoEncoder(inner_model, lr=p.LR)
    conv_model.load_from_checkpoint(
        artifact_dir+"/model.ckpt", model=inner_model)
    conv_model.train()
    conv_model.to(torch.device("cuda:"+str(p.GPU_ID)))
    # conv_model.freeze()
    # for param in conv_model.autoencoder.encoder.parameters():
    #     param.requires_grad = False
    # conv_model.autoencoder.encoder.eval()
    return conv_model.autoencoder


class OneDAdapterAutoencoder(torch.nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = torch.nn.Identity()
        self.decoder = torch.nn.Identity()

    def forward(self, x):
        return self.decoder(self.encoder(x))


if __name__ == "__main__":
    # outer_model = load_outer_model(
    #    "niklas-sprengel/MastersThesis/model-s1w3vien:v0")
    outer_model = models.Fast1DConvModel(
        nChannels=p.CHANNELS, bottleneck_size=13, H=128, W=128)

    model = models.LitAutoEncoder(models.FastCombinedGeneralHyperprior(
        nChannels=p.CHANNELS, outerModel=outer_model, bottleneck_size=13,
        innerModel=models.GeneralHyperprior(main_autoencoder=OneDAdapterAutoencoder(),
                                            hyperprior_autoencoder=models.ChengHyperprior())),
                                  # loss=metrics.RateDistortionLoss(
                                  #    lmbda=p.RATE_DISTORTION_LDMBA),
                                  loss=metrics.MSELossWithBPPEstimation(),
                                  lr=p.LR,
                                  model_type=models.ModelType.CONV_1D_AND_2D_WITH_HYPERPRIOR)

    data.train_and_test(model, do_summary=True)
