import numpy as np

import torch

def convert_to_one_hot_encode(tensor, n_classes):
    """
    Converts a tensor of integer labels to a tensor of one-hot encoded labels.
    
    Parameters:
        tensor (torch.Tensor): The tensor of integer labels.
        n_classes (int): The number of classes in the classification task.
        
    Returns:
        torch.Tensor: The tensor of one-hot encoded labels.
    """
    np_array = tensor.cpu().numpy()
    one_hot = np.empty([np_array.shape[0], n_classes, np_array.shape[1], np_array.shape[2]])

    for i in range(n_classes):
        one_hot[:, i, :, :] = np_array == i
        
    return torch.tensor(one_hot.astype(float))