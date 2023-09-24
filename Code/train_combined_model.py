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
    inner_model = models.Conv1DModel(
        nChannels=p.CHANNELS, num_poolings=4)
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


if __name__ == "__main__":
    # Old latent 13
    # model_id = "3gm16mbp:v1"
    # New latent 13
    model_id = "2urbamfy:v0"
    # New latent 26
    #model_id = "3grh8utl:v0"
    outer_model = load_outer_model(
        f"niklas-sprengel/MastersThesis/model-{model_id}")

    model = models.LitAutoEncoder(models.CombinedModel(
        nChannels=p.CHANNELS, innerChannels=26, H=128, W=128, outerModel=outer_model),
        lr=p.LR, loss=metrics.DualMSELoss(p.DUAL_MSE_LOSS_LMBDA), model_type=models.ModelType.CONV1D_AND_2D)

    data.train_and_test(model)
