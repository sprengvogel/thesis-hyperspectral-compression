import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv1DModel(
        nChannels=p.CHANNELS, num_poolings=7), lr=p.LR)
    # model.load_from_checkpoint(
    #    "MastersThesis/1oj2tot0/checkpoints/last.ckpt", model=model.autoencoder)
    data.train_and_test(model)
