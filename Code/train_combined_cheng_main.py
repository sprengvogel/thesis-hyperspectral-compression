import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p
import torch
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

    model = models.LitAutoEncoder(models.CombinedModel(
        nChannels=p.CHANNELS, innerChannels=13, outerModel=outer_model,
        innerModel=models.ChengMain(in_channels=13)), lr=p.LR,
        loss=metrics.DualMSELoss(p.DUAL_MSE_LOSS_LMBDA),
        model_type=models.ModelType.CONV1D_AND_2D)

    data.train_and_test(model, do_summary=True)
