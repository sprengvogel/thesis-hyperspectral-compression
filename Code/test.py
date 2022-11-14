import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
import torch

if __name__ == "__main__":
    test_model = models.Conv1DModel(nChannels=369)
    summary(test_model, input_size=(1000, 369))
    exit(0)
    model = models.LitSimpleModel()
    summary(model.autoencoder, input_size=(16, 369, 96, 96))

    dataset = data.MatDataset("hypercomp/data/mat-data/")
    dataloader = data.dataLoader(dataset)
    for el in dataloader:
        print(el.shape)

    wandb_logger = WandbLogger(project="MastersThesis")
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    trainer = pl.Trainer(
        accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=1)
    trainer.fit(model, dataloader)
