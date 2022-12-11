# Input Channels
CHANNELS = 369
# Learning rate
LR = 1e-4
LR_HYPERPRIOR = 1e-6
# Shuffle Data loaders? Should be false for debugging
SHUFFLE_DATA_LOADER = False
# Number of workers for data loaders
NUM_WORKERS = 0
# Batch size for all data loaders
BATCH_SIZE = 1
# Special batch size for hyperprior net
BATCH_SIZE_HYPERPRIOR = 16
# Batch size for conv 2D net
BATCH_SIZE_CONV2D = 8
# Number of epochs for training
EPOCHS = 50
# Folders for data
#DATA_FOLDER = "/media/storagecube/data/datasets/hyperspectral/fatih/data"
DATA_FOLDER = "/home/niklassp/data"
#DATA_FOLDER = "/faststorage/fatih-dataset/data"
DATA_FOLDER_SQUIRREL = "/media/storagecube/data/datasets/fatih/squirrel/full/"
