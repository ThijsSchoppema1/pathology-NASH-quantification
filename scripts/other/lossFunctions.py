# Loss functions used by train_model.py

import torch.nn.functional as F
from torch import nn
import torch
from other.soft_n_cut_loss import soft_n_cut_loss
from other.soft_n_cut_loss_class import NCutLoss2D

def Wnet_loss(inputs, segmentations, reconstructions, patch_size):
    l_snc = soft_n_cut_loss(inputs, segmentations, (patch_size, patch_size))
    l_recon = reconstruction_loss(inputs, reconstructions)
    return l_recon + l_snc

def reconstruction_loss(inputs, reconstruction):
    binary_cross_entropy = F.mse_loss(reconstruction, inputs, reduction='sum')
    return binary_cross_entropy

def sep_losses(lweights=None):
    sfncl = NCutLoss2D()
    return sfncl, reconstruction_loss

def getSupervisedLoss(lweights=None):
    # return [nn.BCEWithLogitsLoss()]
    # return [nn.NLLLoss()]
    # return [nn.CrossEntropyLoss(weight=torch.tensor((0.1, 0.2, 0.7)))]
    if lweights:
        return [nn.CrossEntropyLoss(weight=torch.tensor(lweights))]
    return [nn.CrossEntropyLoss()]