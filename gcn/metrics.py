import torch
import torch.nn.functional as F
from torch import nn as nn
from torch.autograd import Variable

class NewMetric:
    def __init__(self, params, **kwargs):
        super(NewMetric, self).__init__()
        pass
    def __call__(self, inputs, targets):
        pass

class Accuracy:
    def __init__(self, **kwargs):
        pass
    def __call__(self, inputs, targets):
        inputs = F.softmax(inputs)
        labels = torch.argmax(inputs, dim = 1)
        targets = targets.long()
        accuracy = torch.mean((labels == targets).to(torch.double)).detach().cpu()
        return accuracy

class Correlation:
    def __init__(self, **kwargs):
        super(Correlation, self).__init__()
        pass
    def __call__(self, inputs, targets):
        corr = F.cosine_similarity(inputs.squeeze() - inputs.mean(), targets - targets.mean(), dim=0)
        return corr
class ClassMSE:
    def __init__(self, ratio=3, base=6, **kwargs):
        super(ClassMSE, self).__init__()
        self.ratio = ratio
        self.base = base
    def __call__(self, inputs, targets):
        targets = targets - self.base
        predict_reg = (F.softmax(inputs) * self.ratio * torch.arange(inputs.shape[-1]).float().to(inputs.device)).sum(dim = -1)        
        reg_loss = F.mse_loss(predict_reg.float(), targets)
        return reg_loss

class ClassCorr:
    def __init__(self, ratio=3, base=6, **kwargs):
        super(ClassCorr, self).__init__()
        self.ratio = ratio
        self.base = base
    def __call__(self, inputs, targets):
        targets = targets - self.base
        predict_reg = (F.softmax(inputs) * self.ratio * torch.arange(inputs.shape[-1]).float().to(inputs.device)).sum(dim = -1)
        corr = F.cosine_similarity(predict_reg - predict_reg.mean(), targets - targets.mean(), dim=0)
        return corr

def get_evaluation_metric(name):
    if name == 'BCEWithLogitsLoss':
        return nn.BCEWithLogitsLoss()
    elif name == 'CrossEntropyLoss':
        return nn.CrossEntropyLoss()
    elif name == 'MSELoss':
        return nn.MSELoss()
    elif name == 'ClassMSE':
        return ClassMSE()
    elif name == 'ClassCorr':
        return ClassCorr()
    elif name == 'Accuracy':
        return Accuracy()
    elif name == 'Correlation':
        return Correlation()
    else:
        raise RuntimeError(f"Unsupported metric function: '{name}'.")