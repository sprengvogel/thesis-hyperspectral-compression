import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.ChengMain(
        in_channels=p.CHANNELS), lr=p.LR)
    data.train_and_test(model, batch_size=p.BATCH_SIZE_CONV2D)
