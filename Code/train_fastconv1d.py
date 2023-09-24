import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p


if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Fast1DConvModel(
        nChannels=202, H=128, W=128, bottleneck_size=13), lr=p.LR)
    data.train_and_test(model, batch_size=4)
