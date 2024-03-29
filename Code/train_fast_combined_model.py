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
    conv_model.freeze()
    # for param in conv_model.autoencoder.encoder.parameters():
    #    param.requires_grad = False
    # conv_model.autoencoder.encoder.eval()
    return conv_model.autoencoder


if __name__ == "__main__":
    # Old standard (latent 13)
    # model_id = "37bkqy6m:v0"
    # Bitrate comp latent 26
    # model_id = "2lnyd5cg:v0"
    # Bitrate comp latent 13
    # model_id = "s1w3vien:v0"
    # Bitrate comp latent 6
    # model_id = "3dyoflur:v0"
    # Bitrate comp latent 3
    model_id = "3begwj6c:v0"
    # Half training set
    # model_id = "3i8bkxdb:v0"
    # eight training set
    # model_id = "1j1kzin7:v0"
    # 16 training set
    # model_id = "1i01q906:v0"
    # 32 training set
    # model_id = "jnb4pfjc:v0"
    # 64 training set
    # model_id = "1k4b8jzb:v0"
    # 128 training set
    # model_id = "2k8qdqvk:v0"
    outer_model = load_outer_model(
        f"niklas-sprengel/MastersThesis/model-{model_id}")

    model = models.LitAutoEncoder(models.FastCombinedModel(
        nChannels=p.CHANNELS, bottleneck_size=3, H=128, W=128, outerModel=outer_model,
        kernel_size=p.KERNEL_SIZE_CONV2D, use_groups=p.USE_GROUPS),
        lr=p.LR, loss=metrics.DualMSELoss(p.DUAL_MSE_LOSS_LMBDA), model_type=models.ModelType.CONV1D_AND_2D)
    data.train_and_test(
        model, use_early_stopping=p.USE_EARLY_STOPPING, batch_size=4)
