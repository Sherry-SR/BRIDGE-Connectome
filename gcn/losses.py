import torch
import torch.nn.functional as F
from torch import nn as nn
import numpy as np

class NewLoss(nn.Module):
    def __init__(self, params):
        super(NewLoss, self).__init__()
        pass
    def forward(self, inputs,targets):
        pass

class GaussianKLDLoss(nn.Module):
    def __init__(self):
        super(GaussianKLDLoss, self).__init__()
        pass
    
    def forward(self, logvar, mu):
        KLDLoss = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / (logvar.size(1))
        KLDLoss = KLDLoss / logvar.size(0)
        return KLDLoss

class RecLoss(nn.Module):
    def __init__(self):
        super(RecLoss, self).__init__()
        pass
    
    def forward(self, outputs, targets):
        outputs = outputs.view(outputs.size(0), -1)
        targets = targets.view(targets.size(0), -1)
        RecLoss = 1 - 2. * (outputs * targets).sum(dim=1) / (outputs + targets).sum(dim=1)
        return RecLoss.mean()*10

class MSEClassLoss(nn.Module):
    def __init__(self, ratio=3, base=6, class_counts = None, **kwargs):
        super(MSEClassLoss, self).__init__()
        self.ratio = ratio
        self.base = base
        if class_counts is not None:
            if isinstance(class_counts, str):
                class_counts = 1 / np.load(class_counts)
                class_counts = torch.tensor(class_counts / np.mean(class_counts), dtype=torch.float32)
            if isinstance(class_counts, list):
                class_counts = 1 / np.array(class_counts)
                class_counts = torch.tensor(class_counts / np.mean(class_counts), dtype=torch.float32)
        self.class_counts = class_counts

    def forward(self, inputs, targets):
        targets = targets - self.base
        if self.class_counts is not None:
            CE_loss = F.cross_entropy(inputs, torch.floor(targets / self.ratio).long(), self.class_counts.to(inputs.device))
        else:
            CE_loss = F.cross_entropy(inputs, torch.floor(targets / self.ratio).long())
        predict_reg = (F.softmax(inputs) * self.ratio * torch.arange(inputs.shape[-1]).float().to(inputs.device)).sum(dim = -1)
        reg_loss = F.mse_loss(predict_reg, targets)
        return CE_loss + reg_loss

def get_loss_criterion(name):
    if name == 'GaussianKLDLoss':
        return GaussianKLDLoss()
    elif name == 'RecLoss':
        return RecLoss()
    elif name == 'BCELoss':
        return nn.BCELoss()
    elif name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'MSEClassLoss':
        return MSEClassLoss()
    elif name == 'SmoothL1Loss':
        return nn.SmoothL1Loss()
    elif name == 'L1Loss':
        return nn.L1Loss()
    else:
        raise RuntimeError(f"Unsupported loss function: '{name}'")