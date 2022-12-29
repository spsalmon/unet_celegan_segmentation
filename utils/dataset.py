import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import skimage.io as skio
import os
import scipy.ndimage as si
from skimage.exposure import equalize_adapthist
import tifffile as ti
import multiprocessing as mp


class DifferentChannelNumberException(Exception):
    """Raise for when the images in the directory have a different number of channels."""

def get_number_of_classes_img(img):
    return np.max(ti.imread(img)) + 1

def get_number_of_channels_img(img):
    return ti.imread(img).shape[0]

def get_number_of_channels_database(dir): 
    imgs = [os.path.join(dir, x) for x in os.listdir(dir)]
    with mp.Pool(32) as pool:
        nb_channels = pool.map(get_number_of_channels_img, imgs)
    if np.min(nb_channels) != np.max(nb_channels):
        raise DifferentChannelNumberException("Error ! The images in the directory have different numbers of channels.")
    else:
        return int(nb_channels[0])

# DATASET FOR TRAINING
class TrainingDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, n_classes = None, method = "binary", dim=512):
        assert (method == "binary" or method == "semantic"), "Unknown method, should be 'binary' or 'semantic'"
        try:
            self.imgs_dir = imgs_dir
            self.masks_dir = masks_dir

            self.n_channels = get_number_of_channels_database(imgs_dir)
            if method == "binary":
                self.n_classes = 1
            elif method == "semantic":
                assert n_classes is not None, "Method is 'semantic' and no number of classes were given. Use the argument -c or --classes."
                self.n_classes = n_classes
            self.dim = dim

            list_id = []
            for file in os.listdir(imgs_dir):
                if not file.startswith('.'):
                    list_id.append(file)

            self.ids = list_id
            logging.info(f'Creating dataset with {len(self.ids)} examples')
        except DifferentChannelNumberException:
            raise
    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, augment_contrast=True):
        if augment_contrast == True:
            improved_img = np.empty_like(img, dtype="float64")
            improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
            improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

            scale_x = dim/img.shape[1]
            scale_y = dim/img.shape[2]

            new_img = si.zoom(improved_img, zoom=[1, scale_x, scale_y])
            new_img = new_img.transpose((0, 1, 2)).astype('float64')

        else:
            normalized_img = np.empty_like(img, dtype="float64")
            normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
            normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

            scale_x = dim/img.shape[1]
            scale_y = dim/img.shape[2]

            new_img = si.zoom(normalized_img, zoom=[1, scale_x, scale_y])
            new_img = new_img.transpose((0, 1, 2)).astype('float64')
        return new_img

    @classmethod
    def preprocess_mask(cls, img, dim):
        scale_x = dim/img.shape[0]
        scale_y = dim/img.shape[1]
        new_mask = si.zoom(img, zoom=[scale_x, scale_y])
        new_mask = np.expand_dims(new_mask, axis=2).transpose(
            (2, 0, 1)).astype('float64')
        return new_mask

    def __getitem__(self, i):
        idx = self.ids[i]

        img_file = [self.imgs_dir + idx]
        mask_file = [self.masks_dir + "seg_" + idx]
        # mask_file = [self.masks_dir + idx]

        mask = skio.imread(mask_file[0], plugin="tifffile")
        img = ti.imread(img_file[0])

        assert img.size//self.n_channels == mask.size, \
            f'Image and mask {idx} should be the same size, but are {img.size} and {mask.size}'

        img = self.preprocess_img(img, self.dim).astype('float64')
        mask = self.preprocess_mask(mask, self.dim).astype('float64')

        return {'image': torch.tensor(img), 'mask': torch.tensor(mask)}


# DATASET FOR PREDICTION

class PredictionDataset(Dataset):
    def __init__(self, imgs_dir, n_classes, dim=512):
        self.imgs_dir = imgs_dir
        self.n_channels = get_number_of_channels_database(imgs_dir)
        self.n_classes = n_classes
        self.dim = dim

        list_id = []
        for file in os.listdir(imgs_dir):
            if not file.startswith('.'):
                list_id.append(file)

        self.ids = sorted(list_id)
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, augment_contrast=True):
        if augment_contrast == True:
            improved_img = np.empty_like(img, dtype="float64")
            improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
            improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

            scale_x = dim/img.shape[1]
            scale_y = dim/img.shape[2]

            new_img = si.zoom(improved_img, zoom=[1, scale_x, scale_y])
            new_img = new_img.transpose((0, 1, 2)).astype('float64')

        else:
            normalized_img = np.empty_like(img, dtype="float64")
            normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
            normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

            scale_x = dim/img.shape[1]
            scale_y = dim/img.shape[2]

            new_img = si.zoom(normalized_img, zoom=[1, scale_x, scale_y])
            new_img = new_img.transpose((0, 1, 2)).astype('float64')
        return new_img

    def __getitem__(self, i):

        idx = self.ids[i]

        img_file = [self.imgs_dir + idx]

        img = ti.imread(img_file[0])
        im_shape = img.shape

        img = self.preprocess_img(img, self.dim).astype('float64')

        return {'image': torch.tensor(img), 'img_path': img_file, 'im_shape': np.array(im_shape)}
