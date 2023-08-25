# An implementation of the W-net, not working as expected see w_net2 for a working implementation

import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

def select_layers(norm_method, dropout, out_dim):
    if norm_method == "batch_norm":
        norm_layer = nn.BatchNorm2d(out_dim)
    elif norm_method == "instance_norm":
        norm_layer = nn.InstanceNorm2d(out_dim)
    elif norm_method == None:
        norm_layer = None

    if dropout:
        dropout_layer = nn.Dropout(dropout)
    else:
        dropout_layer = None

    return norm_layer, dropout_layer

class ConvModule(nn.Module):
    def __init__(self, in_dim, out_dim, norm_method="batch_norm", dropout=0.3, ):
        super(ConvModule, self).__init__()

        norm_layer, dropout_layer = select_layers(norm_method, dropout, out_dim)
        
        layers = [
            nn.Conv2d(in_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim),
            norm_layer,
            nn.ReLU(),
            dropout_layer,
            nn.Conv2d(out_dim, out_dim, 1),
            nn.Conv2d(out_dim, out_dim, 3, padding=1, groups=out_dim),
            norm_layer,
            nn.ReLU(),
            dropout_layer,
        ]

        self.conv_mod = nn.Sequential(*[layer for layer in layers if layer])

    def forward(self, x):
        return self.conv_mod(x)
        
class UNet(nn.Module):
    def __init__(
        self, 
        in_dim, # The channels
        out_dim, # The channels
        norm_method,
        dropout, 
        encoder_in_sizes, 
        decoder_in_sizes,
        last_layer_size=64
        ):
        super(UNet, self).__init__()

        # Create first and last modules
        norm_layer, dropout_layer = select_layers(norm_method, dropout, 64)
        first_layers = [
            nn.Conv2d(in_dim, encoder_in_sizes[0], 3, padding=1),
            norm_layer,
            nn.ReLU(),
            dropout_layer,

            nn.Conv2d(encoder_in_sizes[0], encoder_in_sizes[0], 3, padding=1),
            norm_layer,
            nn.ReLU(),
            dropout_layer,        
        ]
        last_layers = [
            nn.Conv2d(encoder_in_sizes[0]*3, last_layer_size, 3, padding=1),
            norm_layer,
            nn.ReLU(),
            dropout_layer,

            # nn.Conv2d(last_layer_size, last_layer_size, 3, padding=1),
            # norm_layer,
            # nn.ReLU(),
            # dropout_layer, 

            nn.Conv2d(last_layer_size, out_dim, 1), # No padding on pointwise
            nn.ReLU(),
        ]
        
        self.first_module = nn.Sequential(*[layer for layer in first_layers if layer])
        self.last_module = nn.Sequential(*[layer for layer in last_layers if layer])
        self.pool = nn.MaxPool2d(2, 2)

        # Encoder modules
        self.enc_modules = nn.ModuleList(
            [ConvModule(channels, 2*channels) for channels in encoder_in_sizes])

        decoder_out_sizes = [int(x/2) for x in decoder_in_sizes]
        self.dec_transpose_layers = nn.ModuleList(
            [nn.ConvTranspose2d(channels, channels, 2, stride=2) for channels in decoder_in_sizes])
        self.dec_modules = nn.ModuleList(
            [ConvModule(3*channels_out, channels_out) for channels_out in decoder_out_sizes])
        self.last_dec_transpose_layer = nn.ConvTranspose2d(encoder_in_sizes[0]*2, encoder_in_sizes[0]*2, 2, stride=2)
    
    def forward(self, x):
        module_outputs = [self.first_module(x)]
        for module in self.enc_modules:
            module_outputs.append(module(self.pool(module_outputs[-1])))

        # Stole this again
        x_ = module_outputs.pop(-1)
        for conv, upconv in zip(self.dec_modules, self.dec_transpose_layers):
            skip_connection = module_outputs.pop(-1)
            x_ = conv(
                torch.cat((skip_connection, upconv(x_)), 1)
            )
        
        y_hat = self.last_dec_transpose_layer(x_)
        y_hat = torch.cat((module_outputs[-1], y_hat), 1)
        y_hat = self.last_module(y_hat)

        return y_hat
        

class WNet(nn.Module):
    def __init__(
        self,
        in_dim,
        out_dim,
        norm_method,
        dropout,
        encoder_in_sizes=[64, 128, 256],
        decoder_in_sizes=[512, 256],
        last_layer_size=64
        ):
        super(WNet, self).__init__()

        self.U_enc = UNet(
            in_dim=in_dim, 
            out_dim=out_dim,
            norm_method=norm_method,
            dropout=dropout,
            encoder_in_sizes=encoder_in_sizes,
            decoder_in_sizes=decoder_in_sizes,
            last_layer_size=last_layer_size
        )

        self.U_dec = UNet(
            in_dim=out_dim, 
            out_dim=in_dim,
            norm_method=norm_method,
            dropout=dropout,
            encoder_in_sizes=encoder_in_sizes,
            decoder_in_sizes=decoder_in_sizes,
            last_layer_size=last_layer_size
        )
        
        self.softmax = nn.Softmax2d()
        self.sigmoid = nn.Sigmoid()

    def forward_encoder(self, x):
        y_hat = self.U_enc(x)
        return self.softmax(y_hat)

    def forward_decoder(self, x):
        y_hat = self.U_dec(x)
        return self.sigmoid(y_hat)

    def forward(self, x):
        segmentation = self.forward_encoder(x)
        reconstruction = self.forward_decoder(segmentation)
        return segmentation, reconstruction
    
    def remove_dec(self):
        self.U_dec = None

    def return_enc(self):
        return self.U_enc