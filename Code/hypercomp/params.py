# Input Channels
CHANNELS = 202
# Learning rate
LR = 1e-5
LR_HYPERPRIOR = 1e-6
# Specific optimizer parameters
WEIGHT_DECAY = 0
DUAL_MSE_LOSS_LMBDA = 1.0
# Shuffle Data loaders? Should be false for debugging
SHUFFLE_DATA_LOADER = False
# Number of workers for data loaders
NUM_WORKERS = 0
# Batch size for all data loaders
BATCH_SIZE = 1
# Special batch size for hyperprior net
BATCH_SIZE_HYPERPRIOR = 16
# Batch size for conv 2D net
BATCH_SIZE_CONV2D = 4
# Number of epochs for training
EPOCHS = 70
# Folders for data
#DATA_FOLDER = "/media/storagecube/data/datasets/hyperspectral/fatih/data"
DATA_FOLDER = "/home/niklassp/data"
#DATA_FOLDER = "/faststorage/fatih-dataset/data"
DATA_FOLDER_SQUIRREL = "/media/storagecube/data/datasets/fatih/squirrel/full/"
DATA_FOLDER_HYSPECNET = "/media/storagecube/data/datasets/enmap/dataset"
# ID of gpu to train on (taken from nivida-smi)
GPU_ID = 2

KEYWORDS = "spatial, spectral, learned hyperspectral image compression, analyzing latent space, (transformer, cnn)"
