# Input Channels
CHANNELS = 202
# Learning rate
LR = 5e-5
LR_HYPERPRIOR = 1e-6
# Specific optimizer parameters
WEIGHT_DECAY = 0
DUAL_MSE_LOSS_LMBDA = 1.0
RATE_DISTORTION_LDMBA = 0.1
# Early stopping is only implemented for FastCombinedModel
USE_EARLY_STOPPING = True
# Kernel size for conv2d model (only odd allowed)
KERNEL_SIZE_CONV2D = 3
# Use a group per channel in the spatial autoencoder
USE_GROUPS = True
# Shuffle Data loaders? Should be false for debugging
SHUFFLE_DATA_LOADER = True
# Number of workers for data loaders
NUM_WORKERS = 8
# Batch size for all data loaders
BATCH_SIZE = 1
# Special batch size for hyperprior net
BATCH_SIZE_HYPERPRIOR = 1
# Batch size for conv 2D net
BATCH_SIZE_CONV2D = 1
# Number of epochs for training
EPOCHS = 50
# Folders for data
# DATA_FOLDER = "/media/storagecube/data/datasets/hyperspectral/fatih/data"
DATA_FOLDER = "/home/niklassp/data"
# DATA_FOLDER = "/faststorage/fatih-dataset/data"
DATA_FOLDER_SQUIRREL = "/media/storagecube/data/datasets/fatih/squirrel/full/"
DATA_FOLDER_HYSPECNET = "/mnt/data/enmap/dataset"
# ID of gpu to train on (taken from nivida-smi)
GPU_ID = 0

KEYWORDS = "spatial, spectral, learned hyperspectral image compression, analyzing latent space, (transformer, cnn)"
