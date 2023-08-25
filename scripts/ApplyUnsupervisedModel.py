# Deprecated, better to use the fast-inference package with the customProcessors/wnet_processor.py
# Script to apply the unsupervised WNET model

import click
from pathlib import Path

from models.lichting_module import LitModelTrainer
import pytorch_lightning as pl
import torch
from models.w_net2 import WNet
from torch.utils.data import DataLoader

from dataProcesses import dataGenerators
from dataProcesses import augmentations
import multiresolutionimageinterface as mir
from functools import partial
import numpy as np

import imageio
import numpy as np
import cv2 as cv

@click.command()
@click.option('--in_dir', type=Path)
@click.option('--out_dir', type=Path)
@click.option('--color_type', type=click.Choice(['HE_L', 'PSR']))
@click.option('--level', type=int, default=0)
@click.option('--modelpath', type=Path)
def main(
    in_dir,
    out_dir,
    color_type,
    level,
    modelpath
    ):

    if '.ckpt' == modelpath.suffix:
        model = LitModelTrainer.load_from_checkpoint(modelpath)
        model.setup_inference()
    elif '.pt' == modelpath.suffix:
        model = WNet(
            3,
            64,
            "batch_norm",
            0.2,
        )
        model.load_state_dict(torch.load(modelpath))
    else:
        print("using untrained model")
        model = WNet(
            3,
            64,
            "batch_norm",
            0.3,
        )

        model = LitModelTrainer(model, (None, None), None, 64, 224, False)

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    print('We are using the {} device.'.format(device))
    model = model.to(device)
    model.eval()

    in_files = [f for f in in_dir.iterdir() if f.is_file()]
    for in_file in in_files:
        if color_type in in_file.stem:

            out_file = out_dir / in_file.stem
            out_file.mkdir(parents=False, exist_ok=True)
            in_file = str(in_file)

            print(f'### Start model')
            print(f'# in_file={in_file}')
            applyModel(in_file, out_file, level, model, device)
            return
    return

def applyModel(in_file, out_file, level, model, device):
    data_gen = dataGenerators.WSIInferenceSet(in_file, level, patch_size=(256, 256), device=device)
    patchNr = data_gen.getPatchNr()

    size = len(data_gen)
    predarr = []

    print(size, patchNr)
    halved = 0
    splits=4
    with torch.no_grad():
        for idx in range(len(data_gen)):
            patch = data_gen[idx]
            preds = model(patch)
            
            if len(preds) == 2:
                preds = preds[0]
                
            
            preds = preds.detach().cpu().numpy()
            predarr.append(np.copy(preds))

            if idx % (size // 10) == 0:
                print(f'# at {idx} of {size}', preds.dtype)

            if len(predarr) / patchNr[0] == patchNr[1] // splits and halved != 3:
                halved += 1
                file_name = f'split{halved}_result'
                write_results(predarr, out_file, file_name, patchNr[0], patchNr[1] // splits, model.n_classes)
                predarr = []
                return

    if halved != 0:
        file_name = f'split{halved+1}_result'
    else:
        file_name = 'result'
    write_results(predarr, out_file, file_name, patchNr[0], patchNr[1] - ((patchNr[1] // splits)*halved), model.n_classes)
    return

def write_results(predarr, outfile, file_name, x, y, n_classes):

    predarr = [np.concatenate(predarr[(i*(x)):((i+1)*x)], axis=2) for i in range(y)]
    predarr = np.concatenate(predarr, axis=1)

    unique, counts = np.unique(predarr, return_counts=True)
    print(len(unique), len(counts), predarr.shape)
    for u, c in zip(unique, counts):
        print(f"Cluster {u} : Found {c}")

    print(f'# Write as col map')
    predarr = np.moveaxis(predarr, 0, -1)
    predarr = cv.applyColorMap(predarr.astype(np.uint8), cv.COLORMAP_JET)
    imageio.imwrite(outfile / (file_name +'2.png'), predarr.squeeze())
    
    print(f'# Write as png')
    predarr = (predarr / n_classes * 255).astype(np.uint8)
    imageio.imwrite(outfile / (file_name + '.png'), predarr.squeeze())
    return

if __name__ == "__main__":
    main()
