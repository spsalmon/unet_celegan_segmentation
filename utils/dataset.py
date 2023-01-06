import numpy as np
import torch
from torch.utils.data import Dataset
import logging
import os
import scipy.ndimage as ndi
from skimage.exposure import equalize_adapthist
import tifffile as ti
import multiprocessing as mp


class DifferentChannelNumberException(Exception):
    """Raise for when the images in the directory have a different number of channels."""


def get_number_of_channels_img(img: str) -> int:
    """Returns the number of channels in the given image.

    Parameters:
        img (str): path to the image file

    Returns:
        int: number of channels in the image
    """
    img_array = ti.imread(img)
    if len(img_array.shape) == 2:
        img_array = np.expand_dims(img_array, 0)
    return img_array.shape[0]


def get_number_of_channels_database(dir: str) -> int:
    """
    This function returns the number of channels of the images in a directory.
    If the images in the directory have different number of channels, a DifferentChannelNumberException is raised.

    Parameters:
    dir (str): The path to the directory containing the images.

    Returns:
    int: The number of channels of the images.

    Raises:
    DifferentChannelNumberException: If the images in the directory have different number of channels.
    """
    imgs = [os.path.join(dir, x) for x in os.listdir(dir)]
    with mp.Pool(32) as pool:
        nb_channels = pool.map(get_number_of_channels_img, imgs)
    if np.min(nb_channels) != np.max(nb_channels):
        raise DifferentChannelNumberException(
            "Error ! The images in the directory have different numbers of channels.")
    else:
        return int(nb_channels[0])


class TrainingDataset(Dataset):
    """A PyTorch dataset for training a segmentation model.

    Args:
        imgs_dir (str): Path to the directory containing the training images.
        masks_dir (str): Path to the directory containing the corresponding masks.
        n_classes (int, optional): Number of classes for the segmentation. Required if method is set to 'semantic'.
        method (str, optional): Segmentation method, should be either "binary" or "semantic". Defaults to "binary".
        dim (int, optional): Downscaling factor of the images. Defaults to 512.
    """

    def __init__(self, imgs_dir, masks_dir, n_channels = None, n_classes=None, method="binary", dim=512):
        assert (method == "binary" or method ==
                "semantic"), "Unknown method, should be 'binary' or 'semantic'"
        try:
            self.imgs_dir = imgs_dir
            self.masks_dir = masks_dir

            # Get the number of channels in the images in the directory
            if n_channels is None:
                self.n_channels = get_number_of_channels_database(imgs_dir)

            if method == "binary":
                self.n_classes = 1
            elif method == "semantic":
                assert n_classes is not None, "Method is 'semantic' and no number of classes were given. Use the argument -c or --classes."
                self.n_classes = n_classes
            self.dim = dim

            self.ids = [x for x in os.listdir(
                imgs_dir) if not x.startswith('.')]

            logging.info(f'Creating dataset with {len(self.ids)} examples')
        except DifferentChannelNumberException:
            raise

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, n_channels, augment_contrast=True):
        """
        Preprocesses the image to be fed into the model. Optionally augments the contrast of the image.
        Scales the image down to the specified dimensions.

        Parameters:
            img (np.ndarray): The image to be preprocessed.
            dim (int): The target dimension for the image.
            augment_contrast (bool): If True, increases the contrast of the image. Defauls to True.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if n_channels == 2:
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
                improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                    img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img
            
        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[0, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

    @classmethod
    def preprocess_mask(cls, mask, dim: int):
        """Resize and normalize the mask image to the specified dimensions.

        Parameters:
        mask (ndarray): The image mask to be preprocessed.
        dim (int): The desired dimensions of the preprocessed mask image.

        Returns:
        ndarray: The preprocessed mask image.
        """
        # Resize the mask
        scale_x = dim / mask.shape[0]
        scale_y = dim / mask.shape[1]
        resized_mask = ndi.zoom(mask, zoom=[scale_x, scale_y])

        # Add a channel dimension
        processed_mask = np.expand_dims(resized_mask, axis=0).astype('float64')

        return processed_mask

    def __getitem__(self, i):

        # Get the id of the image and mask
        idx = self.ids[i]

        # Get the filepaths of the image and mask
        img_file = [self.imgs_dir + idx]
        mask_file = [self.masks_dir + "seg_" + idx]

        # Read the mask and image
        mask = ti.imread(mask_file[0])
        img = ti.imread(img_file[0])

        assert (img.shape[1], img.shape[2]) == mask.shape, \
            f'Image and mask {idx} should be the same size, but are {img.shape[0]} and {mask.shape}'

        # Preprocess the image and mask
        img = self.preprocess_img(img, self.dim, self.n_channels).astype('float64')
        mask = self.preprocess_mask(mask, self.dim).astype('float64')

        return {'image': torch.tensor(img), 'mask': torch.tensor(mask)}


# DATASET FOR PREDICTION

class PredictionDataset(Dataset):
    """A PyTorch dataset for predicting segmentation masks.

    Args:
        imgs_dir (str): Path to the directory containing the training images.
        n_classes (int): Number of classes for the segmentation.
        dim (int, optional): Downscaling factor of the images. Defaults to 512.
    """
    def __init__(self, imgs_dir, n_classes, n_channels = None, dim=512):
        self.imgs_dir = imgs_dir

        if n_channels is None:
            self.n_channels = get_number_of_channels_database(imgs_dir)
            
        self.n_classes = n_classes
        self.dim = dim

        self.ids = [x for x in os.listdir(imgs_dir) if not x.startswith('.')]
        logging.info(f'Creating dataset with {len(self.ids)} examples')

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess_img(cls, img, dim, n_channels, augment_contrast=True):
        """
        Preprocesses the image to be fed into the model. Optionally augments the contrast of the image.
        Scales the image down to the specified dimensions.

        Parameters:
            img (np.ndarray): The image to be preprocessed.
            dim (int): The target dimension for the image.
            augment_contrast (bool): If True, increases the contrast of the image. Defauls to True.

        Returns:
            np.ndarray: The preprocessed image.
        """
        if n_channels == 2:
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.0005, kernel_size=int(img[0, :, :].shape[0]/16))
                improved_img[1, :, :] = equalize_adapthist(img[1, :, :], nbins=np.max(
                    img[1, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[1, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])
                normalized_img[1, :, :] = img[1, :, :]/np.max(img[1, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img
            
        elif n_channels == 1:
            img = np.expand_dims(img, 0)
            if augment_contrast == True:
                # Augment the image's contrast
                improved_img = np.empty_like(img, dtype="float64")
                improved_img[0, :, :] = equalize_adapthist(img[0, :, :], nbins=np.max(
                    img[0, :, :]) - 1, clip_limit=0.005, kernel_size=int(img[0, :, :].shape[0]/16))

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(improved_img, zoom=[1, scale_x, scale_y])

                processed_img = processed_img.transpose(
                    (0, 1, 2)).astype('float64')

            else:
                # Normalize the image
                normalized_img = np.empty_like(img, dtype="float64")
                normalized_img[0, :, :] = img[0, :, :]/np.max(img[0, :, :])

                # Resize the image
                scale_x = dim/img.shape[1]
                scale_y = dim/img.shape[2]
                processed_img = ndi.zoom(normalized_img, zoom=[
                                        1, scale_x, scale_y])

                processed_img = processed_img.astype('float64')
            return processed_img

    def __getitem__(self, i):
        # Get the id of the image
        idx = self.ids[i]
        img_file = self.imgs_dir + idx

        # Read the image
        img = ti.imread(img_file)
        im_shape = img.shape

        # Preprocess the image
        img = self.preprocess_img(img, self.dim, self.n_channels).astype('float64')

        return {'image': torch.tensor(img), 'img_path': img_file, 'im_shape': np.array(im_shape)}
