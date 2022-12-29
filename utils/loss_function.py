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

class BinaryFocalTverskyLoss(nn.Module):
    def __init__(self, weight=None, smooth=1, alpha=0.7, gamma = 2):
        super(BinaryFocalTverskyLoss, self).__init__()
        self.weight = weight
        self.smooth = smooth
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, predict, target):
        assert predict.shape == target.shape, 'predict & target shape do not match'
        probs = torch.sigmoid(predict)

        probs = probs.view(-1)
        target = target.view(-1)
        
        #True Positives, False Positives & False Negatives
        TP = (probs * target).sum()    
        FP = ((1-target) * probs).sum()
        FN = (target * (1-probs)).sum()
        
        Tversky = (TP + self.smooth) / (TP + (1-self.alpha)*FP + self.alpha*FN + self.smooth)  
        FocalTversky = (1 - Tversky)**self.gamma
                       
        return FocalTversky

# class BinaryFocalLoss(nn.Module):
#     def __init__(self, weight=None, gamma=2, alpha=0.85, ignore_index=None, smooth=None):
#         super(BinaryFocalLoss, self).__init__()
#         self.weight = weight
#         self.gamma = gamma
#         self.alpha = alpha
#         self.ignore_index = ignore_index
#         self.smooth = smooth

#     def forward(self, predict, target):
#         assert predict.shape == target.shape, 'predict & target shape do not match'
#         eps = 1e-15
#         eps_tensor = torch.ones_like(predict)*eps
#         p = torch.maximum(torch.sigmoid(predict), eps_tensor)
#         q = torch.maximum(1-p, eps_tensor)
#         pos_los = -self.alpha*(q ** self.gamma)*torch.log(p)
#         neg_los = -(1-self.alpha)*(p ** self.gamma)*torch.log(q)
#         if self.smooth is None : 
#             target = target.type(torch.bool)
#             loss = torch.where(target, pos_los, neg_los)
        
#         return loss.mean()

class BinaryFocalLoss(nn.Module):
    def __init__(self, weight=None, size_average=True):
        super(BinaryFocalLoss, self).__init__()

    def forward(self, inputs, targets, alpha=0.8, gamma=2, smooth=1):
        
        #comment out if your model contains a sigmoid or equivalent activation layer
        inputs = torch.sigmoid(inputs)       
        
        #flatten label and prediction tensors
        inputs = inputs.view(-1)
        targets = targets.view(-1)
        
        #first compute binary cross-entropy 
        BCE = F.binary_cross_entropy(inputs, targets, reduction='mean')
        BCE_EXP = torch.exp(-BCE)
        focal_loss = alpha * (1-BCE_EXP)**gamma * BCE
                       
        return focal_loss

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