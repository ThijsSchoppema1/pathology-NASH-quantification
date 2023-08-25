# Script to extract continious values from the segmentation results
# Best recorderd vallues are sta_area and inf_area_filter2 devided by tissue_area

import click
import numpy as np
import multiresolutionimageinterface as mir
import cv2
import skimage.filters as skf
from scipy import ndimage

from pathlib import Path

@click.command()
@click.option('--level', type=int, default=3, help="Level to process the WSIs at")
@click.option('--in_dir', type=str, help="Path to directory containing the input tif files.")
def main(
        level=3,
        in_dir='./data/results/supervised/eb4_unet_s05_bg_fp4/wsi_preds/corrected'
    ):
    in_dir = Path(in_dir)
    results = [';'.join(['file_name', 'sta_area', 'sta_cluster_count', 'inf_area', 
                         'inf_area_filter1', 'inf_area_filter2', 'inf_loci_filter1',
                          'inf_loci_filter2', 'tissue_area'])]
    for i, in_file in enumerate(in_dir.glob('*.tif')):
        print(f'\nProcessing file nr. {i}: {in_file.stem}')
        try:
            res = process_file(str(in_file), level)
            res = [str(i) for i in res]
            print('Succesfull')
            results.append(';'.join([str(in_file.stem), *res]))
        except:
            print("Error, invalid file")
            error = ['error']*8
            results.append(';'.join([str(in_file.stem), *error]))

    with open(in_dir / f"result_{level}.txt", 'w+') as f:
        print(f'write results at {str(in_dir / f"result_{level}.txt")}')
        f.write('\n'.join(results))

def process_file(in_file, level):
    image_reader = mir.MultiResolutionImageReader()
    mask = image_reader.open(in_file)

    ext_mask = mask.getUCharPatch(0, 0, *mask.getLevelDimensions(level), level)
    mask = None
    
    # Initial est
    preds, counts = np.unique(ext_mask, return_counts=True)
    predDir = {}
    for p, c in zip(preds, counts):
        predDir[p] = c

    print('Fat:', predDir[2] / (predDir[1]+predDir[2]+predDir[3]))
    print('inflammation:', predDir[3] / (predDir[1]+predDir[2]+predDir[3]))
    
    total_area = predDir[1]+predDir[2]+predDir[3]
    sta_p, inf_p = predDir[2], predDir[3]

    # Process staetosis
    sta_c = staetosis(np.copy(ext_mask))

    # Process inflam
    inf_p2, inf_p3, loci_p2, loci_p3, cnts = inflammation(np.copy(ext_mask), total_area)

    # add inflam
    counter = 0
    for c in cnts:
        cv2.fillPoly(ext_mask, [c], 2)
        counter += 1

    return sta_p, sta_c, inf_p, inf_p2, inf_p3, loci_p2, loci_p3, total_area

def staetosis(mask):
    mask[mask == 1] = 0
    mask[mask == 2] = 1
    mask[mask == 3] = 0
    cnts = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print('number of sta loci:', len(cnts))
    cntpixels = []
    for c in cnts:
        pixels = cv2.contourArea(c)
        cntpixels.append(pixels)

    sta_c = len(cntpixels)

    return sta_c

def inflammation(mask, total_area):
    # Remove border foci
    bg_mask = np.copy(mask)
    bg_mask[bg_mask != 0] = 1

    mask[mask == 1] = 0
    mask[mask == 2] = 0
    mask[mask == 3] = 1

    # mask = remove_border(mask, bg_mask)
    preds, counts = np.unique(mask, return_counts=True)

    inf_p2 = 0
    for p, c in zip(preds, counts):
        if p == 1:
            inf_p2 = c
            break

    # Remove small foci
    cnts = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    print('number of loci:', len(cnts))
    cntpixels = []
    for c in cnts:
        pixels = cv2.contourArea(c)
        cntpixels.append(pixels)

    loci_p2 = len(cntpixels)

    thresh = skf.threshold_otsu(np.array(cntpixels))
    cntpixels = [x for x in cntpixels if x >= thresh]

    print('Selected loci:', len(cntpixels))
    print('average size:', np.mean(cntpixels))
    print('average percent:', np.mean([c / total_area for c in cntpixels]))
    print('Threshold:', thresh)

    inf_p3 = np.sum(cntpixels)
    loci_p3 = len(cntpixels)

    return inf_p2, inf_p3, loci_p2, loci_p3, cnts

def remove_border(ext_mask, bg_mask):
    bg_mask = np.array(bg_mask, np.uint8)
    cnts, _ = cv2.findContours(bg_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s = list(ext_mask.shape)
    s[2] = 3
    bimgs = np.zeros(s)
    bimgs = np.array(bimgs, np.uint8)
    cv2.drawContours(bimgs, cnts, -1, (255, 255, 255), 3)
    bimgs = cv2.cvtColor(bimgs, cv2.COLOR_BGR2GRAY)
    bimgs[bimgs!=0]=255
    ext_mask[bimgs == 255] = 2

    # Look for labelled areas which connect with the tissue borders
    labeled_array, num_features = ndimage.label(ext_mask)
    max_vals = ndimage.maximum(ext_mask,labeled_array,range(1,num_features+1))
    m_idx = np.where(max_vals == 2)[0] + 1 

    # remove those areas
    max_index = np.ones(num_features + 1, np.uint8)
    max_index[m_idx] = 0
    ext_mask = ext_mask * max_index[labeled_array]

    return ext_mask

if __name__ == '__main__':
    main()