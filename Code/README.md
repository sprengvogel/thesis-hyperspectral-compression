### Usage Guide

To use the code in this repository, follow the following steps:
1. Create a new mamba (or conda) environment and install the packages given in environment.yml
2. Install the compressai library (this needs to be done with pip): pip install compressai
3. Set hyperparameters in hypercomp/params.py. Most defaults should be set to useful values. You definitely need to set:
    - DATA_FOLDER_HYSPECNET to the path where the hyspecnet-11k dataset is located.
4. Execute a train\_*.py from the Code folder to train that model. On first execution wandb should prompt you for an API key.
If you do not have one, you need to register a wandb account and activate an API key there.

The code was written in a way optimised for fast development changes since the architectures were changing often during development.
For this reason each model type has its own train_*.py file. There are no command line parameters needed for any of these files.
The combined models require a trained conv1d or fastconv1d model to be trained beforehand (depending on the model). To do this, use train_conv1d.py or train_fastconv1d.py. The trained model will have an artifact id in wandb. This artifact id needs to be set as a hyperparameter in hypercomp/params.py (either CURRENT_CONV1D_ARTIFACT or CURRENT_FASTCONV1D_ARTIFACT). 

Each train\_\*.py file runs on the test set after training. The test\_\*.py files are present mostly for historical reasons.
The code is split into modules within the main hypercomp module. The models are defined in hypercomp/models, metrics in hypercomp/metrics and data-related methods in hypercomp/data.
