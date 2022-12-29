import multiprocessing as mp
import numpy as np
import cv2
from skimage import measure, filters, morphology
from skimage.morphology import remove_small_holes
import os
import tifffile as ti
import argparse

pixelsize = 0.65

# def filter_small_objects(mask, area_threshold):

def remove_small_objects(img, min_size=150.):
    # find all your connected components (white blobs in your image)
    nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(img, connectivity=8)
    # connectedComponentswithStats yields every seperated component with information on each of them, such as size
    # the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
    sizes = stats[1:, -1]
    nb_components = nb_components - 1

    # your answer image
    img2 = img
    # for every component in the image, you keep it only if it's above min_size
    for i in range(0, nb_components):
        if sizes[i] < min_size:
            img2[output == i + 1] = 0

    return img2  

def clean_mask(mask, filter_size=3, min_size=422.5/(pixelsize**2)):

    mask = remove_small_objects(mask, min_size)

    mask = filters.median(mask, morphology.disk(filter_size))

    mask = remove_small_holes(mask, 800).astype("uint8")
    return mask.astype("uint8")

def clean_and_save(mask_path):
    print(os.path.join(output_dir, os.path.basename(mask_path)))
    mask = ti.imread(mask_path)

    mask = clean_mask(mask)

    ti.imwrite(os.path.join(output_dir, os.path.basename(mask_path)), mask, compression="zlib")

def get_args():
    parser = argparse.ArgumentParser(description='Clean binary segmentation masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-i', '--input-dir', type=str, required=True,
                        help='Input masks directory', dest='input_dir')
    parser.add_argument('-i', '--output-dir', type=str, required=True,
                        help='Output masks directory', dest='output_dir')
    parser.add_argument('-p', '--pixelsize', type=float, default=0.65,
                        help='Pixel size', dest='pixelsize')
    


    return parser.parse_args()
if __name__ == '__main__':

    args = get_args()
    input_dir = args.input_dir
    output_dir = args.output_dir
    pixelsize = args.pixelsize
    masks = [os.path.join(input_dir, x) for x in os.listdir(input_dir)]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    with mp.Pool(32) as pool:
        pool.map(clean_and_save, masks)





