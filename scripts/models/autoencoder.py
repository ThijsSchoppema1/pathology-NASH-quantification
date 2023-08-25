from models.w_net2 import UNet

import torch
import torch.nn as nn
from torch.autograd import Function

def Simple_Decoder(input_ch, out_ch):
    chs = [input_ch, out_ch]
    conv3d = nn.Conv3d(chs[0], chs[1], kernel_size=3, padding=1)
    layers = [conv3d, torch.nn.Sigmoid()]
    return nn.Sequential(*layers)   

class EnforcePrior(Function):
    def __init__(self, prior):
        self.prior = prior
        self.eps = 1e-8
        
    def forward(self, x):
        # Regions that prior have 0 prob, we dont want to sample from it
        # Adding a very large negative value to the logits (log of the unormalized prob)
        # hopefully prevent that regions from being sample
        self.forbidden_regs = (self.prior < self.eps).float()       
        return x - 1e12*self.forbidden_regs

    def backward(self, grad):
        # Make sure that the forbidden regions have 0 gradiants.
        return grad*(self.forbidden_regs==0).float()

def enforcer(prior, x):
    return EnforcePrior(prior)(x)

class SAE(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm_method,
        dropout,
        bloc_ch,
        prior
        ):
        super(SAE, self).__init__()

        self.U_enc = UNet(
            in_dim=in_dim, 
            out_dim=out_dim,
            norm_method=norm_method,
            dropout=dropout,
            bloc_ch=bloc_ch,
        )

        self.s_dec = Simple_Decoder(out_dim, in_dim)

        self.enforcer = EnforcePrior(prior)

    def forward(self, x):
        return