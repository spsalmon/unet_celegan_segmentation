import argparse
import logging
from lzma import PRESET_DEFAULT
import os
import sys
from xml.etree.ElementPath import prepare_descendant

import numpy as np
import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import AttU_Net
from model import U_Net
from model import R2U_Net
from model import R2AttU_Net
from model import NestedUNet
from utils.loss_function import *
from utils.accuracy import IoU
from utils.dataset import *
from utils.array import *
from scipy import interpolate

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import TrainingDataset
from torch.utils.data import DataLoader, random_split
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
import random
import cv2
from PIL import Image, ImageFilter
from collections import OrderedDict

from pathlib import Path
import skimage
from skimage.morphology import thin, skeletonize

import nvidia_smi

dir_img_train = "/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_database/training_set/raw/"
dir_mask_train = '/mnt/external.data/TowbinLab/spsalmon/shallow_chambers_database/training_set/body_seg/'


def resize_pred_binary(pred, shape):
    if (len(pred.shape) == 4):
        pred = pred.squeeze(0).cpu().numpy()
    else:
        pred = pred.cpu().numpy()

    x, y, z = pred.shape[0], pred.shape[1], pred.shape[2]

    return ndimage.zoom(pred, (x, float(shape[1] / y), float(shape[2] / z)), order=1)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def load_model(model_path, device, n_classes):
    # Load U-Net
    net = AttU_Net(n_channels = 2, n_classes = n_classes)
    # Load Nested Unet
    # net = NestedUNet(n_channels=2, n_classes=1)
    checkpoint = torch.load(model_path, map_location=device)
    net.to(device=device)

    state_dict = checkpoint['model_state_dict']
    # create new OrderedDict that does not contain `module.`
    new_state_dict = OrderedDict()
    for k, v in state_dict.items():
        name = k[7:] # remove `module.`
        new_state_dict[name] = v
    # load params
    
    net.load_state_dict(new_state_dict)

    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
    del checkpoint
    torch.cuda.empty_cache()
    print("Model : epoch = %s | accuracy = %s" % (epoch, iou_val))

    return {'net': net, 'epoch':epoch}

def load_model_parallel(model_path, device, n_classes):
    net = AttU_Net(n_channels=2, n_classes=n_classes)
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']

    epoch = checkpoint['epoch']
    iou_val = checkpoint['iou_val']
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

    return unet


def check_GPU_memory():
    nvidia_smi.nvmlInit()

    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        info = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print("Device {}: {}, Memory : ({:.2f}% free): {}(total), {} (free), {} (used)".format(
            i, nvidia_smi.nvmlDeviceGetName(handle), 100*info.free/info.total, info.total, info.free, info.used))

    nvidia_smi.nvmlShutdown()



def fake_training(net,
                  device,
                  method,
                  epochs=50,
                  batch_size=1,
                  lr=0.1,
                  save_cp=True,
                  save_frequency=10,
                  dim=512):

    # print("###### LOADING NETWORK ######")

    # check_GPU_memory()

    # Load the dataset
    dataset = TrainingDataset(dir_img_train, dir_mask_train,
                              net.module.n_channels, net.module.n_classes, method, dim)
    n_train = len(dataset)
    train_loader = DataLoader(
        dataset, batch_size=batch_size, shuffle=True, num_workers=16, pin_memory=True)

    # Init the optimizer
    optimizer = optim.Adam(net.parameters(), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=1e-4, epochs=epochs, steps_per_epoch=len(train_loader))
    # scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=2, threshold=0.0005)

    # Init the loss function
    if (method == "binary"):
        # criterion = BinaryFocalTverskyLoss()
        criterion = FocalTverskyLoss()
    else:
        criterion = DiceLoss()

    global_step = 0
    # Loop for the epoch

    net.train()

    epoch_loss, epoch_step, epoch_loss_val = 0, 0, 0
    correct_train, element_train, correct_organ, element_organ = 0, 0, 0, 0
    accuracy_val, loss_val, IoU_val = 0, 0, 0

    dataloader_iterator = iter(train_loader)

    # print("###### TRAINING BEGINS ######")
    # check_GPU_memory()

    for i in range(30):
        print("batch", i)
        try:
            batch = next(dataloader_iterator)
        except StopIteration:
            dataloader_iterator = iter(train_loader)
            batch = next(dataloader_iterator)

        print("###### LOADING IMAGES ######")

        imgs = batch['image']
        true_masks = batch['mask']
        assert imgs.shape[1] == net.module.n_channels, \
            f'Network has been defined with {net.module.n_channels} input channels, ' \
            f'but loaded images have {imgs.shape} channels. Please check that ' \
            'the images are loaded correctly.'

        imgs = imgs.to(device=device, dtype=torch.float32)
        mask_type = torch.float32 if (method == "binary") else torch.long
        true_masks = true_masks.to(device=device, dtype=mask_type)

        check_GPU_memory()

        print("###### PREDICTION ######")

        probs = net(imgs)

        del imgs

        check_GPU_memory()

        print("###### BACKPROPAGATION ######")

        # Compute loss function and prediction
        loss = criterion(probs, true_masks)

        predicted = torch.sigmoid(probs).squeeze(1) > 0.5

        del probs

        epoch_loss += float(loss.item())

        # Compute IoU score
        IoU_train = IoU(predicted.cpu().numpy(),
                        true_masks.squeeze(1).cpu().numpy())

        optimizer.zero_grad()
        loss.backward()

        optimizer.step()

        check_GPU_memory()

        print("###### CLEANING ######")

        del loss
        del predicted
        del true_masks

        torch.cuda.empty_cache()

        check_GPU_memory()


def fake_pred(net,
              device,
              method,
              epochs=50,
              batch_size=1,
              lr=0.1,
              save_cp=True,
              save_frequency=10,
              dim=512):

    # print("###### LOADING NETWORK ######")

    # check_GPU_memory()

    dataset_pred = PredictionDataset(dir_img_train, 2, 1, dim=dim)
    pred_loader = DataLoader(dataset_pred, batch_size=batch_size,
                             shuffle=False, num_workers=32, pin_memory=True)

    with torch.no_grad():
        dataloader_iterator = iter(pred_loader)

        for i in range(30):
            print("batch", i)
            try:
                batch = next(dataloader_iterator)
            except StopIteration:
                dataloader_iterator = iter(pred_loader)
                batch = next(dataloader_iterator)

            print("###### LOADING IMAGES ######")

            imgs = batch['image']
            img_paths = batch['img_path'][0]
            img_shapes = batch['im_shape']

            imgs_batch = imgs.to(device=device, dtype=torch.float32)

            check_GPU_memory()

            print("###### PREDICTION ######")

            pred = net(imgs_batch)
            if dataset_pred.method == "binary":
                res = torch.sigmoid(pred) > 0.5
            else:
                res = torch.softmax(pred, dim=1)

            check_GPU_memory()

            del pred

            for i in range(len(img_paths)):
                mask = res[i, :, :, :]
                # print(mask.shape)

                path = img_paths[i]
                shape = img_shapes[i]

                mask = np.squeeze(resize_pred_binary(
                    mask, shape)).astype('uint8')

            print("###### CLEANING ######")
            del imgs
            torch.cuda.empty_cache()
            check_GPU_memory()

    # for batch in train_loader[0:2]:
    #     global_step += 1
    #     epoch_step +=1

    #     imgs = batch['image']
    #     true_masks = batch['mask']
    #     assert imgs.shape[1] == net.module.n_channels, \
    #         f'Network has been defined with {net.module.n_channels} input channels, ' \
    #         f'but loaded images have {imgs.shape} channels. Please check that ' \
    #         'the images are loaded correctly.'

    #     imgs = imgs.to(device=device, dtype=torch.float32)
    #     mask_type = torch.float32 if (method == "binary") else torch.long
    #     true_masks = true_masks.to(device=device, dtype=mask_type)

    #     probs = net(imgs)

    #     # Compute loss function and prediction
    #     loss = criterion(probs, true_masks)
    #     predicted = torch.sigmoid(probs).squeeze(1) > 0.5

    #     epoch_loss += float(loss.item())

    #     # Compute IoU score
    #     IoU_train = IoU(predicted.cpu().numpy(), true_masks.squeeze(1).cpu().numpy())

    #     optimizer.zero_grad()
    #     loss.backward()

    #     optimizer.step()

    #     del loss
    #     del predicted
    #     del probs

check_GPU_memory()

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
# net = AttU_Net(n_channels=2, n_classes=1)
# device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

if torch.cuda.device_count() > 1:
    unet = load_model_parallel("./checkpoints/CP_epoch13.pth", device, 1)
else:
    unet = load_model("./checkpoints/CP_epoch13.pth", device, 1)
unet.to(device=device)

fake_pred(unet, device, "binary", batch_size=6, dim=1056)
