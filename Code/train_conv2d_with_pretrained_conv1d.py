import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from hypercomp.models.utils import unflatten_and_split_apart_batches
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch
import numpy as np
import wandb


@torch.no_grad()
def encodeWithConvModel(x: np.ndarray, conv_model: models.LitAutoEncoder):
    # Model expects batch dimension
    x = np.expand_dims(x, 0)
    x = torch.tensor(x).to(torch.device("cuda:"+str(p.GPU_ID)))
    out = unflatten_and_split_apart_batches(
        conv_model(x)).squeeze()
    # Scale tensor to [0,1]
    out -= out.min()
    out /= out.max()
    return out.detach().cpu().numpy()


if __name__ == "__main__":
    torch.manual_seed(0)
    np.random.seed(0)
    run = wandb.init(project="MastersThesis",
                     job_type="Train_With_Pretrained_Conv")
    artifact = run.use_artifact(
        'niklas-sprengel/MastersThesis/model-1x92mdk2:v0', type='model')
    artifact_dir = artifact.download()
    conv_model = models.LitAutoEncoder(
        models.Conv1DModel(nChannels=369), lr=p.LR)
    conv_model.load_from_checkpoint(artifact_dir+"/model.ckpt")
    conv_model.eval()
    conv_model.freeze()
    conv_model.to(torch.device("cuda:"+str(p.GPU_ID)))

    model = models.LitAutoEncoder(models.Conv2DModel(
        nChannels=93, H=96, W=96), lr=p.LR)
    summary(model.autoencoder, input_size=(p.BATCH_SIZE, 93, 96, 96))

    train_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="train", transform=lambda x: encodeWithConvModel(x, conv_model=conv_model))
    val_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="val", transform=lambda x: encodeWithConvModel(x, conv_model=conv_model))
    test_dataset = data.MatDatasetSquirrel(
        p.DATA_FOLDER_SQUIRREL, split="test", transform=lambda x: encodeWithConvModel(x, conv_model=conv_model))

    train_dataloader = data.dataLoader(
        train_dataset, batch_size=p.BATCH_SIZE_CONV2D)
    val_dataloader = data.dataLoader(
        val_dataset, batch_size=p.BATCH_SIZE_CONV2D)
    test_dataloader = data.dataLoader(
        test_dataset, batch_size=p.BATCH_SIZE_CONV2D)

    wandb_logger = WandbLogger(project="MastersThesis", log_model="all")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID])
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
