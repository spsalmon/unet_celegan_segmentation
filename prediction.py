import argparse
import logging
from lzma import PRESET_DEFAULT
import os
import sys
from xml.etree.ElementPath import prepare_descendant

import numpy as np
import torch
import torch.nn as nn
from tqdm import tqdm

from model import AttU_Net

from utils.loss_function import *
from utils.accuracy import *
from utils.dataset import *
from utils.array import *

from utils.dataset import PredictionDataset
from PIL import Image, ImageFilter
from collections import OrderedDict
from torch.utils.data import DataLoader, random_split

def resize_pred_binary(pred, shape):
    if(len(pred.shape) == 4):
        pred = pred.squeeze(0).cpu().numpy()
    else:
        pred = pred.cpu().numpy()
    
    x, y, z = pred.shape[0], pred.shape[1], pred.shape[2]

    return ndimage.zoom(pred, (x, float(shape[1] / y), float(shape[2] / z)), order=1)

def load_model(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
    dim = checkpoint['dim']
    n_classes = checkpoint['n_classes']
    n_channels = checkpoint['n_channels']

    unet = AttU_Net(n_channels=n_channels, n_classes=n_classes)
    del checkpoint
    torch.cuda.empty_cache()
    print("Model : epoch = %s | accuracy = %s" % (epoch, iou_val))

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    unet.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    del state_dict
    torch.cuda.empty_cache()

    unet.cuda()

    return unet, epoch, dim, n_channels, n_classes, iou_val

def load_model_parallel(model_path, device):
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
    dim = checkpoint['dim']
    n_classes = checkpoint['n_classes']
    n_channels = checkpoint['n_channels']

    net = AttU_Net(n_channels=n_channels, n_classes=n_classes)
    del checkpoint
    torch.cuda.empty_cache()
    print("Model : epoch = %s | accuracy = %s" % (epoch, iou_val))

    new_state_dict = OrderedDict()

    for k, v in state_dict.items():
        name = k[7:]  # remove `module.`
        new_state_dict[name] = v
    net.load_state_dict(new_state_dict)
    logging.info(f'Model loaded from {model_path}')

    del state_dict
    torch.cuda.empty_cache()

    print("Let's use", torch.cuda.device_count(), "GPUs!")
    unet = nn.DataParallel(net)
    unet.cuda()

    return unet, epoch, dim, n_channels, n_classes, iou_val

def get_args():
    parser = argparse.ArgumentParser(description='Predict masks for a set of images',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('-b', '--batch-size', metavar='B', type=int, nargs='?', default=2,
                        help='Batch size', dest='batchsize')
    parser.add_argument('-i', '--images-dir', type=str, required=True,
                        help='Directory containing the input images', dest='raw_dir')
    parser.add_argument('-o', '--output-dir', type=str, required=True,
                        help='Output directory', dest='output_dir')
    parser.add_argument('-m', '--model-path', type=str, required=True,
                        help='Path to the model', dest='model_path')
    return parser.parse_args()

if __name__ == '__main__':

    args = get_args()
    batch_size = args.batchsize
    raw_dir = args.raw_dir
    output_dir = args.output_dir
    model_path = args.model_path


    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # load neural network and define labels

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    if torch.cuda.device_count() > 1:
        unet, epoch, dim, n_channels, n_classes, iou_val = load_model_parallel(model_path, device)
    else : 
        unet, epoch, dim, n_channels, n_classes, iou_val = load_model(model_path, device)

    unet.to(device=device)

    # prediction loop for all the images in the folder. 

    logging.info("Loading of the prediction database")
    dataset_pred = PredictionDataset(raw_dir, n_classes, dim=dim)
    logging.info("Finished loading the prediction database")
    assert n_channels == dataset_pred.n_channels, f"This model was not trained for images with {dataset_pred.n_channels} channels. Either load another model or check your images."

    n_pred = len(dataset_pred)
    pred_loader = DataLoader(dataset_pred, batch_size=batch_size, shuffle=False, num_workers=32, pin_memory=True)

    if n_classes == 1:
        method = "binary"
    else:
        method = "semantic"

    logging.info(f'''Starting prediction:
        Directory:          {raw_dir}
        Model: {model_path}
        Batch size:      {batch_size}
        Classes:   {n_classes}
        Images dimension:  {dim}
        Device:          {device.type}
    ''')

    with tqdm(total=n_pred, desc=f'Prediction', unit='img') as pbar:
        for batch in pred_loader:

            imgs = batch['image']
            img_paths = batch['img_path'][0]
            img_shapes = batch['im_shape']

            with torch.no_grad():
                imgs_batch = imgs.to(device=device, dtype=torch.float32)
                pred = unet(imgs_batch)
                del imgs_batch
                if method == "binary":
                    res = torch.sigmoid(pred) > 0.5
                else:
                    res = torch.softmax(pred, dim=1)
                del pred
                for i in range(len(img_paths)):
                    mask = res[i, :, :, :]
                    
                    path = img_paths[i]
                    shape = img_shapes[i]

                    mask = np.squeeze(resize_pred_binary(mask, shape)).astype('uint8')
                    mask = morphology.binary_closing(mask).astype("uint8")
                    output_file1 = "seg_"+os.path.basename(path)
                    ti.imwrite(os.path.join(output_dir, output_file1), mask, compression ='zlib')
                torch.cuda.empty_cache()
            pbar.update(batch_size)