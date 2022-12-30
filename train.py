import argparse
import logging
import os
import sys
from typing import Tuple

import torch
import torch.nn as nn
from torch import optim
from tqdm import tqdm

from model import AttU_Net
from utils.loss_function import *
from utils.accuracy import *
from utils.dataset import *
from utils.array import *

from torch.utils.tensorboard import SummaryWriter
from utils.dataset import TrainingDataset
from torch.utils.data import DataLoader
from collections import OrderedDict

logging.basicConfig()

# Path where you want your checkpoints saved :
dir_checkpoints = './checkpoints/'


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


def train_net(net,
              device,
              dataset_train,
              dataset_val,
              method="binary",
              epochs=50,
              batch_size=1,
              lr=1e-4,
              save_cp=True,
              save_frequency=10,
              dim=512):

    # Load the datasets

    n_train = len(dataset_train)
    train_loader = DataLoader(
        dataset_train, batch_size=batch_size, shuffle=True, num_workers=32, pin_memory=True)

    n_val = len(dataset_val)
    val_loader = DataLoader(dataset_val, batch_size=batch_size,
                            shuffle=True, num_workers=32, pin_memory=True)

    # Init the tensorboard
    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_DIM_{dim}')
    global_step = 0

    # Print the information
    logging.info(f'''Starting training:
        Epochs:          {epochs}
        Batch size:      {batch_size}
        Learning rate:   {lr}
        Training size:   {n_train}
        Checkpoints:     {save_cp}
        Device:          {device.type}
        Images dim:  {dim}
        Method:          {method}
    ''')

    # Print the information of the graphic card
    if device.type == 'cuda':
        print('Number of gpu : ', torch.cuda.device_count())
        print('Working on : ', torch.cuda.current_device())
        print('Name of the working gpu : ',
              torch.cuda.get_device_name(torch.cuda.current_device()))

    # Init the optimizer
    optimizer = optim.Adam(net.parameters(), lr=lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=lr, epochs=epochs, steps_per_epoch=len(train_loader))

    # Init the loss function
    if (method == "binary"):
        criterion = FocalTverskyLoss()
        dir_checkpoint = dir_checkpoints
    else:
        criterion = DiceLoss()
        dir_checkpoint = dir_checkpoints

    # Loop for the epoch
    for epoch in range(epochs):
        net.train()

        epoch_loss, epoch_step, epoch_loss_val = 0, 0, 0
        accuracy_val, loss_val, IoU_val = 0, 0, 0

        # Init loop for the step
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                global_step += 1
                epoch_step += 1

                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if (
                    method == "binary") else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                probs = net(imgs)

                del imgs

                # Compute loss function and prediction
                loss = criterion(probs, true_masks)
                predicted = torch.sigmoid(probs).squeeze(1) > 0.5

                del probs

                epoch_loss += float(loss.item())
                writer.add_scalar('Loss/train/global',
                                  loss.item(), global_step)
                writer.add_scalar('Loss/train/' + str(epoch+1),
                                  loss.item(), epoch_step)

                # Compute IoU score
                IoU_train = IoU(predicted.cpu().numpy(),
                                true_masks.squeeze(1).cpu().numpy())

                writer.add_scalar('Acc/train/global', IoU_train, global_step)
                writer.add_scalar('Acc/train/' + str(epoch+1),
                                  IoU_train, epoch_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()},
                                 **{'IoU score (train)': IoU_train})

                optimizer.zero_grad()
                loss.backward()

                optimizer.step()

                del loss
                del predicted
                del true_masks

                pbar.update(batch_size)

                torch.cuda.empty_cache()

        print("EPOCH", epoch, "AVERAGE TRAINING LOSS :",
              (epoch_loss/(n_train/batch_size)))

        # This controls how often you're saving a checkpoint and testing your network on the
        # validation set.
        if epoch % save_frequency == 0:
            net.eval()
            with torch.no_grad():
                with tqdm(total=n_val, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
                    for batch in val_loader:
                        imgs = batch['image']
                        true_masks = batch['mask']

                        imgs = imgs.to(device=device, dtype=torch.float32)
                        mask_type = torch.float32
                        true_masks = true_masks.to(
                            device=device, dtype=mask_type)

                        probs = net(imgs)

                        del imgs

                        # Compute loss function and prediction
                        loss = criterion(probs, true_masks)
                        predicted = torch.sigmoid(probs).squeeze(1) > 0.5

                        del probs

                        epoch_loss_val += float(loss.item())
                        writer.add_scalar('Loss/train/global',
                                          loss.item(), global_step)
                        writer.add_scalar(
                            'Loss/train/' + str(epoch+1), loss.item(), epoch_step)

                        # Compute accuracy
                        IoU_batch = float(
                            IoU(predicted.cpu().numpy(), true_masks.squeeze(1).cpu().numpy()))
                        IoU_val += IoU_batch
                        writer.add_scalar('Acc/train/global',
                                          IoU_batch, global_step)
                        writer.add_scalar(
                            'Acc/train/' + str(epoch+1), IoU_batch, epoch_step)

                        pbar.set_postfix(
                            **{'loss (batch)': loss.item()}, **{'IoU score (val)': IoU_batch})

                        pbar.update(batch_size)
                        del loss
                        del predicted
                        del true_masks

                        torch.cuda.empty_cache()

            IoU_val = IoU_val/(n_val/batch_size)
            loss_val = epoch_loss_val/(n_val/batch_size)

            scheduler.step()
            print("EPOCH", epoch, "IoU SCORE ON THE VALIDATION SET :",
                  IoU_val, "AVERAGE LOSS :", loss_val)
            if save_cp:
                try:
                    os.mkdir(dir_checkpoint)
                    logging.info('Created checkpoint directory')
                except OSError:
                    pass
                torch.save({'epoch': epoch + 1,
                            'n_channels': dataset_train.n_channels,
                            'n_classes': dataset_train.n_classes,
                            'dim': dim,
                            'model_state_dict': net.state_dict(),
                            'optimizer_state_dict': optimizer.state_dict(),
                            'loss_val': loss_val,
                            'iou_val': IoU_val,
                            'loss_train': (epoch_loss/(n_train/batch_size))},
                           dir_checkpoint + f'CP_epoch{epoch + 1}.pth')
                logging.info(f'Checkpoint {epoch + 1} saved !')

    writer.close()


def get_args() -> argparse.Namespace:
    """
    Parses the command-line arguments and returns them as a namespace object.

    Returns:
        argparse.Namespace: The namespace object containing the parsed arguments.
    """
    # Create a parser and set the formatter class to ArgumentDefaultsHelpFormatter
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks',
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    
    # Add the arguments to the parser
    parser.add_argument('-e', '--epochs', dest="epochs", type=int, default=21,
                        help='Number of epochs')
    parser.add_argument('-b', '--batch-size', dest="batch_size", type=int, default=1,
                        help='Batch size')
    parser.add_argument('-l', '--learning-rate', dest="lr", type=float, default=1e-4,
                        help='Learning rate')
    parser.add_argument('-f', '--load', dest='load', type=str, default=None,
                        help='Load model from a .pth file')
    parser.add_argument('-d', '--dim', dest='dim', type=int, default=512,
                        help='Downscaling factor of the images')
    parser.add_argument('-m', '--method', dest='method', type=str, default="binary",
                        choices=["binary", "semantic"],
                        help='Segmentation method, should be either "binary" or "semantic"')
    parser.add_argument('-c', '--classes', dest='n_classes', type=int, default=None,
                        help='Use if method is set to "semantic", number of classes for the segmentation.')
    parser.add_argument('-t', '--save-frequency', dest='save_frequency', type=int, default=1,
                        help='Save and test frequency')
    parser.add_argument('--training-dir', dest='training_dir', type=str, required=True,
                        help='Path to the directory containing the training set')
    parser.add_argument('--validation-dir', dest='validation_dir', type=str, required=True,
                        help='Path to the directory containing the validation set')

    # Parse the arguments and return the resulting namespace object
    return parser.parse_args()



if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO,
                        format='%(levelname)s: %(message)s')
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    n_classes = args.n_classes

    # Creating training and validation datasets
    dir_img_train = args.training_dir + "raw/"
    dir_mask_train = args.training_dir + "seg/"

    dir_img_val = args.validation_dir + "raw/"
    dir_mask_val = args.validation_dir + "seg/"

    logging.info(f'Loading training dataset')
    dataset_train = TrainingDataset(
        dir_img_train, dir_mask_train, method=args.method, n_classes=n_classes, dim=args.dim)
    logging.info(f'Finished loading training dataset')

    logging.info(f'Loading validation dataset')
    dataset_val = TrainingDataset(
        dir_img_val, dir_mask_val, method=args.method, n_classes=n_classes, dim=args.dim)
    logging.info(f'Finished loading validation dataset')

    assert dataset_train.n_channels == dataset_val.n_channels, "Training and validation dataset images don't have the same number of channels."

    # Initializing the neural network
    net = AttU_Net(n_channels=dataset_train.n_channels,
                   n_classes=dataset_train.n_classes)

    logging.info(f'Network:\n'
                 f'\t{net.n_channels} input channels\n'
                 f'\t{net.n_classes} output channels (classes)\n')

    if torch.cuda.device_count() > 1:
        if args.load:
            state_dict = torch.load(args.load)['model_state_dict']
            new_state_dict = OrderedDict()

            for k, v in state_dict.items():
                name = k[7:]  # remove `module.`
                new_state_dict[name] = v
            net.load_state_dict(new_state_dict)
            logging.info(f'Model loaded from {args.load}')

        print("Let's use", torch.cuda.device_count(), "GPUs!")
        unet = nn.DataParallel(net)
        unet.cuda()
    else:
        unet = nn.DataParallel(net)
        unet.cuda()
    unet.to(device=device)

    try:
        train_net(net=unet,
                  device=device,
                  epochs=args.epochs,
                  dataset_train=dataset_train,
                  dataset_val=dataset_val,
                  method=args.method,
                  batch_size=args.batch_size,
                  lr=args.lr,
                  save_frequency=args.save_frequency,
                  dim=args.dim)
    except KeyboardInterrupt:
        torch.save(net.state_dict(), 'INTERRUPTED.pth')
        logging.info('Saved interrupt')
        try:
            sys.exit(0)
        except SystemExit:
            os._exit(0)
