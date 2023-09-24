import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv2DModel(
        nChannels=p.CHANNELS, H=128, W=128), lr=p.LR)
    data.train_and_test(model, batch_size=p.BATCH_SIZE_CONV2D)
