import hypercomp.data as data
import hypercomp.models as models
from hypercomp import params as p
from hypercomp import metrics

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.Conv2DWithHyperprior(
        input_channels=p.CHANNELS, H=128, W=128), lr=p.LR, model_type=models.ModelType.HYPERPRIOR,
        loss=metrics.RateDistortionLoss(lmbda=p.RATE_DISTORTION_LDMBA))
    data.train_and_test(model, batch_size=p.BATCH_SIZE_CONV2D)
