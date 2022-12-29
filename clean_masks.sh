#!/bin/bash
#SBATCH -J clean
#SBATCH -o clean.out
#SBATCH -N 1
#SBATCH -c 32
#SBATCH -t 1:00:00
#SBATCH --mem=64G

# TO BE CHANGED TO YOUR INPUT AND OUPUT DIRECTORIES

INPUT_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/quick_test/"
OUTPUT_DIR="/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_analysis_testing/analysis/quick_test/"

# YOU CAN CHANGE THE PIXEL SIZE HERE

PIXELSIZE=0.65

python3 ./toolbox/clean_masks -i "$INPUT_DIR" -o "$OUTPUT_DIR"