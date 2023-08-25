# Script to apply the morphological oprations to the segmentation results

import click
import numpy as np
import multiresolutionimageinterface as mir
import digitalpathology.image.io.imagewriter as img_writer
import digitalpathology.image.io.imagereader as img_reader
import cv2
import skimage.filters as skf
from scipy import ndimage

from pathlib import Path

@click.command()
@click.option('--in_dir', type=Path, help="Path to the directory containing the input tif files.")
@click.option('--out_dir', type=Path, help="Path to the output directory.")
@click.option('--spacing', type=float, default=4.0, help="The spacing to process the WSIs")
@click.option('--overwrite/--no-overwrite', default=False, type=bool, help="Overwrite existing results")
@click.option('--tile_size', default=512, type=int, help="Tile size to process each WSI")
def main(
        in_dir,
        out_dir,
        spacing,
        overwrite,
        tile_size
    ):
    in_dir = Path(in_dir)
    results = [';'.join(['file_name', 'sta_area', 'sta_cluster_count', 'inf_area', 
                         'inf_area_filter1', 'inf_area_filter2', 'inf_loci_filter1',
                          'inf_loci_filter2', 'tissue_area'])]
    for i, in_file in enumerate(in_dir.glob('*.tif')):
        print(f'\nProcessing file nr. {i}: {in_file.stem}')
        try:
            out_path = out_dir / in_file.name
            
            if (not overwrite) and out_path.exists():
                print(f'\nfile exists: {out_path}')
                continue

            mask, mask_file = process_file2(str(in_file), str(out_path), spacing, tile_size)
            
            if mask is None:
                print(f'\nSpacing of {spacing} not found in {mask_file}')
                continue
                    
            if mask is not True:
                write_file(mask, mask_file, str(out_path))
        except:
            print(f"Error, invalid file, {in_file}")

def write_file(mask, mask_file, out_path):
    image_writer = img_writer.ImageWriter(out_path, 
                              mask.shape[0:2], 
                              mask_file.getSpacing(),
                              np.uint8,
                              'indexed',
                              indexed_channels=1,
                              tile_size=max(mask_file.getLevelDimensions(0)))
    image_writer.fill(mask)
    image_writer.close()

def process_file(in_file, spacing):
    image_reader = mir.MultiResolutionImageReader()
    mask = image_reader.open(in_file)
    
    downsample_per_level = [mask.getLevelDownsample(level) for level in range(mask.getNumberOfLevels())]
    level = None
    for i, downsampling in enumerate(downsample_per_level):
        if mask.getSpacing()[0] * downsampling == spacing:
            level = i
            break
    if level is None:
        return None, downsample_per_level
    
    ext_mask = mask.getUCharPatch(0, 0, *mask.getLevelDimensions(level), level)
    ext_mask[ext_mask >= 4] = 4
    
    bg_mask = np.zeros_like(ext_mask).astype(np.uint8)
    bg_mask[ext_mask != 0] = 1
    tissue_sum = np.sum(bg_mask)
    bg_mask = None

    # Process portals
    ext_mask = portals(ext_mask, spacing, tissue_sum)

    # Process Inflam
    ext_mask = inflam(ext_mask, spacing, tissue_sum)

    # Process Stae
    ext_mask = steatosis(ext_mask, spacing, tissue_sum)

    return ext_mask, mask

def process_file2(in_file, out_path, spacing, tile_size):
    image_reader = img_reader.ImageReader(in_file)
    
    if not image_reader.test(spacing):
        spacings = image_reader.spacings()
        image_reader.close()
        return None, spacings
    level = image_reader.level(spacing)
    shape = image_reader.shapes[level]
    
    temp_image_reader = mir.MultiResolutionImageReader()
    full_img = temp_image_reader.open(in_file)
    full_img = full_img.getUCharPatch(0, 0, *full_img.getLevelDimensions(level), level)
    full_img[full_img != 0] = 1
    tissue_sum = np.sum(full_img)
    full_img = None
    temp_image_reader = None
    
    image_writer = img_writer.ImageWriter(out_path, 
                              shape, 
                              spacing,
                              np.uint8,
                              'indexed',
                              indexed_channels=1,
                              tile_size=tile_size)
    
    
    rows = -(-shape[0] // tile_size)
    cols = -(-shape[1] // tile_size)
    
    for r in range(0, shape[0], tile_size):
        for c in range(0, shape[1], tile_size):
            patch = image_reader.read(spacing, r, c, tile_size, tile_size)
            patch = process_patch(patch, spacing, tissue_sum)
            image_writer.write(patch, r, c)
            
    image_reader.close()
    image_writer.close()

    return True, True

def process_patch(ext_mask, spacing, tissue_sum):
    ext_mask[ext_mask >= 4] = 4
    
    # kernel = 1
    if spacing == 4.0:
        spacing = 6.0
    if spacing == 4.0:
        spacing = 1.0
    # if spacing == 0.5:
    #     kernel = 40
    
    
    
    # Process portals
    ext_mask = portals(ext_mask, spacing, tissue_sum)

    # Process Inflam
    ext_mask = inflam(ext_mask, spacing, tissue_sum)

    # Process Stae
    ext_mask = steatosis(ext_mask, spacing, tissue_sum)

    return ext_mask
    

def portals(ext_mask, spacing, tissue_sum):
    bin_mask = np.zeros_like(ext_mask).astype(np.uint8)
    bin_mask[ext_mask == 4] = 1
    
    # Morph operations
    kernel = np.ones((int(20 // spacing),int(20 // spacing)),np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations = 10)
    bin_mask = bin_mask.astype(np.uint8)

    # remove Small hits
    # cnts = cv2.findContours(bin_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # cntpixels = []
    # for c in cnts:
    #     pixels = cv2.contourArea(c)
    #     cntpixels.append(pixels)
        
    # if len(cntpixels) != 0:
    #     thresh = skf.threshold_otsu(np.array(cntpixels))
    #     for cnt, cntpixel in zip(cnts, cntpixels):
    #         if cntpixel < thresh and cntpixel/tissue_sum < 0.01 :
    #             cv2.fillPoly(bin_mask, [cnt], 0)

    # ext_mask[ext_mask == 4] = 1
    # ext_mask[bin_mask == 1] = 4
    
    return ext_mask

def inflam(ext_mask, spacing, tissue_sum):
    bin_mask = np.zeros_like(ext_mask).astype(np.uint8)
    bin_mask[ext_mask == 3] = 1

    # Morph operations
    kernel = np.ones((int(20 // spacing),int(20 // spacing)),np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    bin_mask = bin_mask.astype(np.uint8)

    # Remove border hits
    bg_mask = np.zeros_like(ext_mask).astype(np.uint8)
    bg_mask[ext_mask != 0] = 1

    kernel = np.ones((int(20 // spacing),int(20 // spacing)),np.uint8)
    bg_mask = cv2.morphologyEx(bg_mask, cv2.MORPH_CLOSE, kernel)
    bg_mask = cv2.erode(bg_mask, kernel,iterations = 5)

    bin_mask[bg_mask == 0] = 0
    bg_mask = None

    # Remove portal hits
    p_bin_mask = np.zeros_like(ext_mask).astype(np.uint8)
    p_bin_mask[ext_mask == 4] = 1
    p_bin_mask = cv2.dilate(p_bin_mask, kernel,iterations = 5)
    bin_mask[p_bin_mask == 1] = 0
    p_bin_mask = None

    # labeled_array, num_features = ndimage.label(bin_mask)
    # for i in range(num_features):
    #     try:
    #         max_val = np.amax(bin_mask[labeled_array == i])
    #     except:
    #         max_val = 0
    #     bin_mask[labeled_array == i] = max_val
    # bin_mask[bin_mask == 2] = 0

    # remove Small hits
    # cnts = cv2.findContours(bin_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    # cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    # cntpixels = []
    # for c in cnts:
    #     pixels = cv2.contourArea(c)
    #     cntpixels.append(pixels)

    # if len(cntpixels) != 0:
    #     thresh = skf.threshold_otsu(np.array(cntpixels))
    #     for cnt, cntpixel in zip(cnts, cntpixels):
    #         if cntpixel/tissue_sum < 0.001 :
    #             cv2.fillPoly(bin_mask, [cnt], 0)

    # apply bin mask
    ext_mask[ext_mask == 3] = 1
    ext_mask[bin_mask == 1] = 3

    return ext_mask

def steatosis(ext_mask, spacing, tissue_sum):
    bin_mask = np.zeros_like(ext_mask).astype(np.uint8)
    bin_mask[ext_mask == 2] = 1
    
    # Morph operations
    kernel = np.ones((int(5 // spacing),int(5 // spacing)),np.uint8)
    bin_mask = cv2.morphologyEx(bin_mask, cv2.MORPH_OPEN, kernel)
    # mask = cv2.dilate(mask, kernel, iterations = 10)
    bin_mask = bin_mask.astype(np.uint8)

    # apply bin mask
    ext_mask[ext_mask == 2] = 1
    ext_mask[bin_mask == 1] = 2

    return ext_mask

if __name__ == '__main__':
    main()