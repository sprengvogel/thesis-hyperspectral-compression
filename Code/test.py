import hypercomp.data as data
import hypercomp.models as models
import pytorch_lightning as pl
from hypercomp import params as p
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger

if __name__ == "__main__":
    model = models.LitSimpleModel()
    summary(model.autoencoder, input_size=(16, 369, 96, 96))

    dataset = data.MatDataset("hypercomp/data/mat-data/")
    dataloader = data.dataLoader(dataset)
    for el in dataloader:
        print(el.shape)

    wandb_logger = WandbLogger(project="MastersThesis")
    trainer = pl.Trainer(
        accelerator="gpu", max_epochs=p.EPOCHS, logger=wandb_logger)
    trainer.fit(model, dataloader)
