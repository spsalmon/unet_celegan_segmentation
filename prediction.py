import argparse
import logging
import os
from typing import Tuple

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm
import skimage.morphology as morphology
import scipy.ndimage as ndi

from model import AttU_Net

from utils.loss_function import *
from utils.accuracy import *
from utils.dataset import *
from utils.array import *

from utils.dataset import PredictionDataset
from torch.utils.data import DataLoader
from collections import OrderedDict

logging.basicConfig(level=logging.NOTSET)

def resize_prediction(prediction: torch.Tensor, shape: Tuple) -> np.ndarray:
    """
    Resizes the given prediction to the specified shape.

    Parameters:
        prediction (torch.Tensor): The prediction to resize.
        shape (tuple): The desired shape of the resized prediction.

    Returns:
        numpy.ndarray: The resized prediction.
    """
    # # If the prediction has 4 dimensions, remove the first dimension (batch size) and convert to numpy array
    # if len(prediction.shape) == 4:
    #     prediction = prediction.squeeze(0).cpu().numpy()
    # else:
    #     prediction = prediction.cpu().numpy()

    # Convert the prediction to a numpy array
    prediction = prediction.cpu().numpy()

    # Extract the x, y, and z dimensions of the prediction
    pred_x, pred_y, pred_z = prediction.shape[0], prediction.shape[1], prediction.shape[2]

    # Return the resized prediction using ndi's zoom function with order 1
    return ndi.zoom(prediction, (pred_x, float(shape[1] / pred_y), float(shape[2] / pred_z)), order=1)


def load_model(model_path: str, device: torch.device) -> Tuple[nn.Module, int, int, int, int, float]:
    """
    Loads the model from the given path and returns it, along with other relevant information.

    Parameters:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: A tuple containing the following elements:
            - nn.Module: The loaded model.
            - int: The number of epochs the model was trained for.
            - int: The size of the model input.
            - int: The number of channels in the model input.
            - int: The number of classes in the model output.
            - float: The IoU (intersection over union) value for the model.
    """
    # Load the checkpoint from the given path and extract relevant information
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
    dim = checkpoint['dim']
    n_classes = checkpoint['n_classes']
    n_channels = checkpoint['n_channels']

    # Create an instance of the Attention UNet model
    unet = AttU_Net(n_channels=n_channels, n_classes=n_classes)

    # Create a new ordered dictionary to store the state dictionary without the `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    # Move model to the chosen device
    unet.to(device=device)

    # Return network and relevant information about it
    return unet, epoch, dim, n_channels, n_classes, iou_val


def load_model_parallel(model_path: str, device: torch.device) -> Tuple[nn.Module, int, int, int, int, float]:
    """
    Loads the model from the given path and returns it as a DataParallel model, along with other relevant information.

    Parameters:
        model_path (str): The path to the model checkpoint.
        device (torch.device): The device to load the model onto.

    Returns:
        tuple: A tuple containing the following elements:
            - nn.Module: The loaded model wrapped in a DataParallel module.
            - int: The epoch at which the model was trained.
            - int: The size of the model input.
            - int: The number of channels in the model input.
            - int: The number of classes in the model output.
            - float: The IoU (intersection over union) value for the model.
    """
    print("Let's use", torch.cuda.device_count(), "GPUs!")
    # Load the checkpoint from the given path and extract relevant information
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
    dim = checkpoint['dim']
    n_classes = checkpoint['n_classes']
    n_channels = checkpoint['n_channels']

    # Create an instance of the Attention UNet model
    unet = AttU_Net(n_channels=n_channels, n_classes=n_classes)

    # Create a new ordered dictionary to store the state dictionary without the `module.` prefix
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    del state_dict
    torch.cuda.empty_cache()

    # Wrap the model in a DataParallel module
    unet = nn.parallel.DataParallel(unet)

    # Move model to the chosen device
    unet.to(device=device)

    # Return network and relevant information about it
    return unet, epoch, dim, n_channels, n_classes, iou_val


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
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, default=6,
                        help='Batch size', dest='batch_size')
    parser.add_argument('-i', '--images-dir', type=str, required=True,
                        help='Directory containing the input images', dest='raw_dir')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory', dest='output_dir')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to the model', dest='model_path')

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()

def predict(raw_dir: str, output_dir:str, model_path: str, batch_size: int, device: torch.device) -> None:
    """
    Makes predictions for the images in the given directory using the model at the given path.
    
    Parameters:
        raw_dir (str): The directory containing the input images.
        model_path (str): The path to the model.
        batch_size (int): The batch size to use for prediction.
        device (torch.device): The device to run the prediction on.
    """
    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Load the model
    if torch.cuda.device_count() > 1:
        unet, epoch, dim, n_channels, n_classes, iou_val = load_model_parallel(
            model_path, device)
    else:
        unet, epoch, dim, n_channels, n_classes, iou_val = load_model(
            model_path, device)

    # Load the prediction dataset
    logging.info("Loading of the prediction database")
    dataset_pred = PredictionDataset(raw_dir, n_classes, dim=dim)
    logging.info("Finished loading the prediction database")
    assert n_channels == dataset_pred.n_channels, f"This model was not trained for images with {dataset_pred.n_channels} channels. Either load another model or check your images."

    # Create a data loader for the prediction dataset
    n_pred = len(dataset_pred)
    pred_loader = DataLoader(dataset_pred, batch_size=batch_size,
                             shuffle=False, num_workers=32, pin_memory=True)

    # Determine the prediction method based on the number of classes
    if n_classes == 1:
        method = "binary"
    else:
        method = "semantic"

    logging.info(f'''Starting prediction:
        Directory:          {raw_dir}
        Model: {model_path}
        Epochs : {epoch}
        IoU on validation set : {iou_val}
        Batch size:      {batch_size}
        Classes:   {n_classes}
        Images dimension:  {dim}
        Device:          {device.type}
    ''')

    # Prediction loop for all the images in the folder
    with tqdm(total=n_pred, desc=f'Prediction', unit='img') as pbar:
        with torch.no_grad():
            for batch in pred_loader:
            # Extract the images, image paths, and shapes from the batch
                imgs = batch['image']
                img_paths = batch['img_path'][0]
                img_shapes = batch['im_shape']

                # Move the images to the device and convert them to float
                imgs_batch = imgs.to(device=device, dtype=torch.float32)

                # Make the prediction
                pred = unet(imgs_batch)

                # Delete the images from the GPU to free up memory
                del imgs_batch

                # Apply sigmoid or softmax on prediction depending on the method
                if method == "binary":
                    res = torch.sigmoid(pred) > 0.5
                else:
                    res = torch.softmax(pred, dim=1)

                # Delete the prediction from the GPU to free up memory
                del pred

                # Loop over the images in the batch
                for i in range(len(img_paths)):
                    # Extract the mask and relevant information for the current image
                    mask = res[i, :, :, :]
                    path = img_paths[i]
                    shape = img_shapes[i]

                    # Resize the mask and apply a binary closing operation
                    mask = np.squeeze(resize_prediction(mask, shape)).astype('uint8')
                    mask = morphology.binary_closing(mask).astype("uint8")

                    # Save the mask to the output directory
                    output_file1 = "seg_" + os.path.basename(path)
                    ti.imwrite(os.path.join(output_dir, output_file1), mask, compression='zlib')

                # Delete the prediction result from the GPU to free up memory
                del res
                # Free up GPU memory
                torch.cuda.empty_cache()
                # Update the progress bar
                pbar.update(batch_size)


if __name__ == '__main__':

    args = get_args()
    batch_size = args.batch_size
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    model_path = args.model_path

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Run prediction
    predict(raw_dir, output_dir, model_path, batch_size, device)