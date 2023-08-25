# Remove the second U-net from the W-net

import torch
import os
import click
from pathlib import Path

# os.chdir('./pathology-NASH-quantification')

import sys
sys.path.insert(1, './pathology-NASH-quantification')

from scripts.models.w_net2 import WNet, UNet
from scripts.models.lichting_module import LitModelTrainer

@click.command()
@click.option('--modelpath', type=Path)
def main(modelpath):

    if '.ckpt' == modelpath.suffix:
        model = LitModelTrainer.load_from_checkpoint(modelpath)
        model = model.get_model()
    elif '.pt' == modelpath.suffix:
        model = WNet(
            3,
            64,
            "batch_norm",
            0.2,
        )
        model.load_state_dict(torch.load(modelpath))

    model = model.return_enc()

    torch.save(model.state_dict(), modelpath.parents[0] / 'model_enc_statedict.pt')
    
if __name__ == "__main__":
    main()