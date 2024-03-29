import torch
import pytorch_lightning as pl
from ..models import LitAutoEncoder
from .hySpecNet11k import HySpecNet11k
from torchinfo import summary
from pytorch_lightning.loggers import WandbLogger
from pytorch_lightning.callbacks import ModelCheckpoint, EarlyStopping
from torch.utils.data import Dataset, DataLoader
from .. import params as p


def dataLoader(dataset: Dataset, batch_size=p.BATCH_SIZE, shuffle=p.SHUFFLE_DATA_LOADER) -> DataLoader:
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=p.NUM_WORKERS, pin_memory=True, drop_last=True)


def train_and_test(model: LitAutoEncoder, batch_size=p.BATCH_SIZE, do_summary=True, use_early_stopping=False):
    pl.seed_everything(0)
    if do_summary:
        summary(model.autoencoder, input_size=(batch_size,
                                               p.CHANNELS, 128, 128), device="cuda:"+str(p.GPU_ID))
    # else:
    #     model.example_input_array = torch.zeros(
    #         1, 202, 128, 128, device=torch.device("cuda:"+str(p.GPU_ID)))
    #     model.to(torch.device("cuda:"+str(p.GPU_ID)))
    #     print(ModelSummary(model, max_depth=5))
    #     return

    train_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="train")
    val_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="val")
    test_dataset = HySpecNet11k(
        p.DATA_FOLDER_HYSPECNET, mode="easy", split="test")

    train_dataloader = dataLoader(train_dataset, batch_size=batch_size)
    val_dataloader = dataLoader(
        val_dataset, batch_size=batch_size, shuffle=False)
    test_dataloader = dataLoader(
        test_dataset, batch_size=batch_size, shuffle=False)

    wandb_logger = WandbLogger(project="MastersThesis", log_model=True)
    checkpoint_callback = ModelCheckpoint(
        save_last=True, save_top_k=1, monitor="val_metrics/psnr", mode="max")
    callbacks = [checkpoint_callback]
    if use_early_stopping:
        early_stopping_callback = EarlyStopping(
            "val_loss/loss", min_delta=5e-8, patience=10, mode="min", check_on_train_epoch_end=False)
        callbacks.append(early_stopping_callback)
    accelerator = "gpu" if torch.cuda.is_available() else "cpu"
    print("Accelerator: " + accelerator)
    trainer = pl.Trainer(gradient_clip_val=1.0,
                         accelerator=accelerator, max_epochs=p.EPOCHS, logger=wandb_logger, log_every_n_steps=50, val_check_interval=1.0, devices=[p.GPU_ID], callbacks=callbacks)
    trainer.fit(model, train_dataloaders=train_dataloader,
                val_dataloaders=val_dataloader)
    trainer.test(model, dataloaders=test_dataloader)
