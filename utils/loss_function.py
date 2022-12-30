import torch
import torch.nn as nn
import torch.nn.functional as F


class DiceLoss(nn.Module):
    def __init__(self, weight=None, ignore_index=None, smooth=1, p=2):
        super(DiceLoss, self).__init__()
        self.weight = weight
        self.ignore_index = ignore_index
        self.smooth = smooth
        self.p = p

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        probs = torch.softmax(predict, dim=1)

        num = 0
        denum = 0 

        for i in range(probs.shape[1]):
            num += torch.mul(probs[:, i, :, :], target[:, i, :, :]).sum()
            denum += torch.pow(probs[:, i, :, :], self.p).sum() + torch.pow(target[:, i, :, :], self.p).sum()
        print(num)
        return 1 - (2 * num / denum)

class FocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(FocalTverskyLoss, self).__init__()

    def forward(self, inputs, targets, smooth=100, alpha=0.3, beta=0.7, gamma = 4/3):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (inputs * targets).sum()    
        FP = ((1-targets) * inputs).sum()
        FN = (targets * (1-inputs)).sum()
       
        Tversky = (TP + smooth) / (TP + alpha*FP + beta*FN + smooth)  

        
        return (1 - Tversky)**(1/gamma)