# Script to extract patches from WSIs to speed up the MIL method

import click
from pathlib import Path
import numpy as np
from PIL import Image

import sys
sys.path.insert(1, './pathology-NASH-quantification/scripts')

from dataProcesses import dataGenerators

@click.command()
@click.option('--in_dir', type=Path)
@click.option('--mask_patern', type=str, default='/maskdir/{image}_tb_mask.tif')
@click.option('--out_par_dir', type=Path)
@click.option('--level', type=int, default=1)
@click.option('--patch_size', type=int, default=224)
def main(in_dir, out_par_dir, mask_patern, patch_size, level):

    if in_dir.is_file():
        in_files = [in_dir]
    else:
        in_files = [f for f in in_dir.iterdir() if f.is_file()]

    out_par_dir.mkdir(parents=True, exist_ok=True)

    for in_file in in_files:
        in_mask = None
        in_mask = mask_patern.format(image=str(in_file.stem))

        out_dir = out_par_dir / in_file.stem

        
        if not Path.exists(out_dir):
            out_dir.mkdir(exist_ok=True)

            in_file = str(in_file)
            print(f'Start file {in_file}')
            process_file(in_file, in_mask, patch_size, level, out_dir)
        else:
            print(f'Skipping file {str(in_file)}')

def process_file(in_file, in_mask, patch_size, level, out_dir):
    data_gen = dataGenerators.WSIInferenceSet(in_file, level, patch_size=(patch_size, patch_size), device='cpu', totensor=False, mask_file=in_mask)

    for idx in range(len(data_gen)):
        patch, mask = data_gen[idx]

        if np.sum(mask) != 0 and np.sum(mask) / (patch_size * 2) >= 0.99:
            im = Image.fromarray(patch)
            im.save(str(out_dir / f"{idx}.png"))

if __name__ == '__main__':
    main()