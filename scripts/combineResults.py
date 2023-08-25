# Script to combine 2 masks from the portal model and inflammation/steatosis model

import click
import numpy as np
import multiresolutionimageinterface as mir
import cv2
import skimage.filters as skf
from scipy import ndimage

from pathlib import Path
import digitalpathology.image.io.imagewriter as img_writer
import digitalpathology.image.io.imagereader as img_reader

@click.command()
@click.option('--mask_dir', type=Path, help="The mask dir containing the lowest spacing")
@click.option('--up_mask_dir', type=Path, help="The mask dir containing the higher spaced masks, the lowest spacing of these masks will be used to as output.")
@click.option('--out_dir', type=Path, help="Path to the output directory.")
@click.option('--overwrite/--no-overwrite', default=False, type=bool, help="Overwrite the existing results.")
@click.option('--patched/--not-patched', default=False, type=bool, help="Parse each file in one go, or use several patches.")
@click.option('--tile_size', default=512, type=int, help="The patch size to use.")
def main(
        mask_dir,
        up_mask_dir,
        out_dir,
        overwrite,
        patched,
        tile_size
    ):
    for i, in_file in enumerate(mask_dir.glob('*.tif')):
        out_file = out_dir / str(in_file.name)
        if (not overwrite) and out_file.exists():
            print(f'\nfile exists: {out_file}')
            continue
        print(f'\nProcessing file nr. {i}: {in_file.stem}')
        print(f'Output file: {out_file}')
        in_file2 = up_mask_dir / str(in_file.name)
        if not patched:
            process_files(str(in_file), str(in_file2), str(out_file))
        else:
            process_files_patched(str(in_file), str(in_file2), str(out_file), tile_size)
    
def process_files(file1, file2, out_file):
    print(file1)
    print(file2)

    image_reader = mir.MultiResolutionImageReader()
    mask = image_reader.open(file1)
    mask2 = image_reader.open(file2)

    if mask is None:
        print(f'Corrupted file, {file1}')
        return
    if mask2 is None:
        print(f'Corrupted file, {file2}')
        return

    spacing2 = mask2.getSpacing()[0]   
    downsample_per_level = [mask.getLevelDownsample(level) for level in range(mask.getNumberOfLevels())]

    level = None
    for i, downsampling in enumerate(downsample_per_level):
        if mask.getSpacing()[0] * downsampling == spacing2:
            level = i
            break

    if level == None:
        print('Error, spacing of {spacing2} not found in {file1}')
        print('Found spacings are {downsample_per_level}')
        return
    
    print('spacing: ', spacing2)
    print('level: ', level)
    print('downsample_per_level: ', downsample_per_level)

    ext_mask = mask.getUCharPatch(0, 0, *mask.getLevelDimensions(level), level)
    ext_mask2 = mask2.getUCharPatch(0, 0, *mask2.getLevelDimensions(0), 0)
    
    ext_mask[ext_mask2==2] = 4
    image_writer = img_writer.ImageWriter(out_file, 
                              ext_mask2.shape[0:2], 
                              mask2.getSpacing(),
                              np.uint8,
                              'indexed',
                              indexed_channels=1,
                              tile_size=max(mask2.getLevelDimensions(0)))
    image_writer.fill(ext_mask)
    image_writer.close()
    
def process_files_patched(file1, file2, out_file, tile_size):
    print(file1)
    print(file2)
    
    x = determine_spacing(file1, file2)
    if x is None:
        return
    
    spacing, level, level2 = x
    
    image_reader1 = img_reader.ImageReader(file1)
    image_reader2 = img_reader.ImageReader(file2)
    
    shape1 = image_reader1.shapes[level]
    shape2 = image_reader2.shapes[level]
    
    if shape1[0] != shape2[0] or shape1[1] != shape2[1]:
        print('Error, different shapes {shape1}, {shape2}')
        image_reader1.close()
        image_reader2.close()
        image_writer.close()  
        return
    
    image_writer = img_writer.ImageWriter(out_file, 
                              shape1, 
                              spacing,
                              np.uint8,
                              'indexed',
                              indexed_channels=1,
                              tile_size=tile_size)
    
    rows = -(-shape1[0] // tile_size)
    cols = -(-shape1[1] // tile_size)
    
    for r in range(0, shape1[0], tile_size):
        for c in range(0, shape1[1], tile_size):
            patch1 = image_reader1.read(spacing, r, c, tile_size, tile_size)
            patch2 = image_reader2.read(spacing, r, c, tile_size, tile_size)
            
            patch1[patch2==2] = 4
            image_writer.write(patch1, r, c)
            
    image_reader1.close()
    image_reader2.close()
    image_writer.close()   
    
def determine_spacing(file1, file2):
    print(file1)
    print(file2)

    image_reader1 = mir.MultiResolutionImageReader()
    image_reader2 = mir.MultiResolutionImageReader()
    mask = image_reader1.open(file1)
    mask2 = image_reader2.open(file2)

    if mask is None:
        print(f'Corrupted file, {file1}')
        return None
    if mask2 is None:
        print(f'Corrupted file, {file2}')
        return None

    spacing2 = mask2.getSpacing()[0]   
    downsample_per_level = [mask.getLevelDownsample(level) for level in range(mask.getNumberOfLevels())]

    level = None
    for i, downsampling in enumerate(downsample_per_level):
        if mask.getSpacing()[0] * downsampling == spacing2:
            level = i
            break

    if level == None:
        print('Error, spacing of {spacing2} not found in {file1}')
        print('Found spacings are {downsample_per_level}')
        return None
    
    print('spacing: ', spacing2)
    print('level: ', level)
    print('downsample_per_level: ', downsample_per_level)
    
    return spacing2, level, 0


if __name__ == '__main__':
    main()