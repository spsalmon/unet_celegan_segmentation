from __future__ import annotations
import shutil
from sklearn.model_selection import train_test_split
import os
import multiprocessing as mp

from itertools import repeat
import argparse
import logging
import tqdm

# Utility function to move images

def move_file_to_folder(f, destination_folder):
    try:
        print(f)
        shutil.copy(f, destination_folder)
    except:
        print(f)
        assert False


def move_files_to_folder(list_of_files, destination_folder, descriptor=""):
    pool = mp.Pool()
    pool.starmap(move_file_to_folder, tqdm.tqdm(zip(
        list_of_files, repeat(destination_folder)), total=len(list_of_files), desc=descriptor))


def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Predict masks for a set of images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    # Add the arguments to the parser
    parser.add_argument('-i', '--images-dir', type=str, required=True,
                        help='Directory containing the training images', dest='raw_dir')
    parser.add_argument('-m', '--masks-dir', type=str, required=True,
                        help='Directory containing the training masks', dest='mask_dir')
    parser.add_argument('-d', '--database-dir', type=str, required=True,
                        help='Output directory for the database', dest='database_dir')
    parser.add_argument('-r', '--ratio', type=float,
                        help='Ratio of data used for testing', dest='ratio', default=0.4)

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()

    raw_dir = args.raw_dir
    seg_dir = args.mask_dir
    database_dir = args.database_dir

    train_dir = database_dir + "training_set/"
    train_raw = train_dir + "raw/"
    train_seg = train_dir + "seg/"

    validation_dir = database_dir + "validation_set/"
    validation_raw = validation_dir + "raw/"
    validation_seg = validation_dir + "seg/"


    test_dir = database_dir + "testing_set/"
    test_raw = test_dir + "raw/"
    test_seg = test_dir + "seg/"


    if not os.path.exists(database_dir):
        os.makedirs(database_dir)

    if not os.path.exists(train_dir):
        os.makedirs(train_dir)

    if not os.path.exists(train_raw):
        os.makedirs(train_raw)

    if not os.path.exists(train_seg):
        os.makedirs(train_seg)


    if not os.path.exists(validation_dir):
        os.makedirs(validation_dir)

    if not os.path.exists(validation_raw):
        os.makedirs(validation_raw)

    if not os.path.exists(validation_seg):
        os.makedirs(validation_seg)

    if not os.path.exists(test_dir):
        os.makedirs(test_dir)

    if not os.path.exists(test_raw):
        os.makedirs(test_raw)

    if not os.path.exists(test_seg):
        os.makedirs(test_seg)


    raws = [os.path.join(raw_dir, x) for x in os.listdir(raw_dir)]
    segs = [os.path.join(seg_dir, x) for x in os.listdir(seg_dir)]

    raws.sort()
    segs.sort()

    # Split the dataset into train-validation-test sets

    # Split between training and testing
    train_raws, testing_raws, train_segs, testing_segs = train_test_split(
        raws, segs, test_size=0.4, random_state=1917)
    # Split the testing set between the validation and test sets
    val_raws, test_raws, val_segs, test_segs = train_test_split(
        testing_raws, testing_segs, test_size=0.5, random_state=1917)

    move_files_to_folder(train_raws, train_raw, "MOVING TRAINING SET RAW IMAGES")
    move_files_to_folder(val_raws, validation_raw, "MOVING VALIDATION SET RAW IMAGES")
    move_files_to_folder(test_raws, test_raw, "MOVING TESTING SET RAW IMAGES")

    move_files_to_folder(train_segs, train_seg, "MOVING TRAINING SET MASKS")
    move_files_to_folder(val_segs, validation_seg, "MOVING VALIDATION SET MASKS")
    move_files_to_folder(train_segs, train_seg, "MOVING TESTING SET MASKS")

