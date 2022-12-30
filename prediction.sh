#!/bin/bash
#SBATCH -J pred
#SBATCH -o pred.out
#SBATCH -N 1
#SBATCH -c 64
#SBATCH -t 48:00:00
#SBATCH --mem=64G
#SBATCH --gres=gpu:1

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

IMAGES_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/ch1_3/"
OUTPUT_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/quick_test/"

# CHANGE FOR THE PATH OF THE MODEL YOU WANT TO USE

MODEL_PATH="./checkpoints/CP_epoch1.pth"

python3 prediction.py -b 6 -i "$IMAGES_DIR" -o "$OUTPUT_DIR" -m "$MODEL_PATH"