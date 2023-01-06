#!/bin/bash
#SBATCH -J split
#SBATCH -o split.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 48:00:00
#SBATCH --mem=64G

# TO BE CHANGED TO YOUR TRAINING AND VALIDATION SET DIRECTORIES

IMAGES_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/ch1_3/"
MASKS_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/ch2_seg/"
DATABASE_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/quick_test/"

# CHANGE FOR THE PATH OF THE MODEL YOU WANT TO USE

RATIO=0.4

python3 ./toolbox/split_sets.py -i "$IMAGES_DIR" -m "$MASKS_DIR" -d "$DATABASE_DIR" -r $RATIO