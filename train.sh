#!/bin/bash
#SBATCH -J train
#SBATCH -o train.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

DATABASE_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_database/"
TRAINING_DIR="${DATABASE_DIR}training_set/"
VALIDATION_DIR="${DATABASE_DIR}validation_set/"

# YOU CAN CHANGE THE LEARNING RATE HERE. BASE VALUE IS 1e-4

LEARNING_RATE=0.0001

# CHANGE THE SEGMENTATION METHOD IF NEEDED

METHOD="binary"

python3 train.py -b 2 -e 31 -t 1 -d 1056 --training-dir "$TRAINING_DIR" --validation-dir "$VALIDATION_DIR" --learning-rate $LEARNING_RATE --method "$METHOD"