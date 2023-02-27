from enum import Enum


class ModelType(Enum):
    HYPERPRIOR = 1
    CONV1D_AND_2D = 2
    OTHER = 3
    VAE = 4
    CONV_1D_AND_2D_WITH_HYPERPRIOR = 5
