# An implementation of the pointwise unet

import torch
import torch.nn as nn
import torch.nn.functional as F

class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, norm_method="batch_norm", dropout=0.3, seperable=True):
        super(ConvModule, self).__init__()

        layers = [
            nn.Conv2d(in_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim),
            nn.InstanceNorm2d(out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim),
            nn.InstanceNorm2d(out_dim),
            nn.BatchNorm2d(out_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        if norm_method == "instance_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if norm_method == "batch_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if dropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.conv_mod = nn.Sequential(*layers)

    def forward(self, x):
        return self.conv_mod(x)
        
class UNet(nn.Module):
    def __init__(
        self, 
        in_dim, # The input channels
        out_dim, # The output classes
        norm_method,
        dropout
        ):
        super(UNet, self).__init__()

        encoder=[64, 128, 256, 512]
        decoder=[1024, 512, 256]
        decoder_out_sizes = [int(x/2) for x in decoder]

        # Pool layer
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder modules
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels, norm_method="batch_norm", dropout=0.3) for channels in encoder])

        # transpose layers
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder]) # Stride of 2 makes it right size
        self.last_dec_transpose_layer = nn.ConvTranspose2d(128, 128, 2, stride=2)

        # Decoder modules
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out, norm_method="batch_norm", dropout=0.3) for channels_out in decoder_out_sizes])

        # First Encoder module
        layers = [
            nn.Conv2d(in_dim, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),
        ]

        if norm_method == "instance_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if norm_method == "batch_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if dropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.first_module = nn.Sequential(*layers)

        # Last Decoder module
        layers = [
            nn.Conv2d(128+64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, 64, 3, padding=1),
            nn.InstanceNorm2d(64),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Dropout(dropout),

            nn.Conv2d(64, out_dim, 1),
        ]

        if norm_method == "instance_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.InstanceNorm2d)]
        if norm_method == "batch_norm":
            layers = [layer for layer in layers if not isinstance(layer, nn.BatchNorm2d)]
        if dropout:
            layers = [layer for layer in layers if not isinstance(layer, nn.Dropout)]

        self.last_module = nn.Sequential(*layers)

    def forward(self, x):
        x1 = self.first_module(x)
        activations = [x1]
        for module in self.enc_modules:
            activations.append(module(self.pool(activations[-1])))

        x_ = activations.pop(-1)

        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = activations.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )

        segmentations = self.last_module(
            torch.cat((activations[-1], self.last_dec_transpose_layer(x_)), 1)
        )
        return segmentations