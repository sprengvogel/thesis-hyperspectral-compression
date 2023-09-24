import hypercomp.data as data
import hypercomp.models as models
import hypercomp.metrics as metrics
from hypercomp import params as p

if __name__ == "__main__":
    model = models.LitAutoEncoder(models.ChengAttentionModel(in_channels=202,
                                                             N=192), lr=p.LR, loss=metrics.RateDistortionLoss(p.RATE_DISTORTION_LDMBA), model_type=models.ModelType.HYPERPRIOR)
    data.train_and_test(model, batch_size = 16, do_summary = False)