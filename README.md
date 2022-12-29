# How to use this code

## How to train

### Prerequisites

* Split your data in two different sets (Training and Validation), one will be used for training your network's parameters, the other one will be used to test the performance of your network as it is training (so you can know when to stop the training and avoid overfitting). 
* Make sure you've installed all the required libraries.

### Training

Modify the folder paths in the **train.py** file.

Create a batch file for running your code on the server (called for example, train.sh) :

#!/bin/bash

python3 /path/segmentation/train.py

Add your desired options depending on what you want and need :

* **-e** : number of epochs (actually, number of epochs minus 1, so if you want 100 epochs, set this to 101)
* **-b** : batch size (how many images will be loaded at the same time during training, ideally you generally want this as big as possible)
* **-l** : learning rate (unless you know what you are doing, or unless there is some problems with the training, network not learning for example, I would recommend to not modify this).
* **-f** : load a .pth (checkpoint file) and start the training from there.
* **-d** : dimension of the downscaled training images (d*d). In order to fit in the memory, your images will have to be downscaled.
* **-t** : save and test frequency (how often the program will save a checkpoint of the network and test it on the validation set), base value is 2, so the network will do it every 2 epochs.

Then use a command similar to this one :

sbatch --mem 64GB --time 72:00:00 --gres=gpu:2 train.sh

I would recommend using 2 GPUs or more for a faster training time.
