# Deprecated, better to use the fast-inference package with the customProcessors/simple_processor.py
#
# Created on 13 Tue 2023
# Author: Thijs Schoppema
#
# Function: Quantify fibrosis, Staetosis and inflamation in WSI
# Fibrosis is being labelled as 1
# Staetosis as 2
# And inflamation as 3

import click
import imageio
from pathlib import Path

import multiresolutionimageinterface as mir
from previousThesis import PSR_segmentaiton, HNE_segmentation
import numpy as np
from dataProcesses import dataGenerators

import skimage.filters as skf

@click.command()
@click.option('--in_dir', type=Path)
@click.option('--in_mask_pat', type=Path, default=None)
@click.option('--out_dir', type=Path)
@click.option('--color_type', type=click.Choice(['HE_L', 'PSR']))
@click.option('--level', type=int, default=1)
@click.option('--partial', type=bool, default=False)
@click.option('--window_size', type=int, default=8192)
def main(
    in_dir,
    in_mask_pat,
    out_dir,
    color_type,
    level,
    partial,
    window_size
    ):
    print("### PARAMETERS ###")
    print(f"in_file:{in_dir}")
    print(f"out_file:{out_dir}")
    print(f"color_type:{color_type}")
    print(f"level:{level}")
    print(f"partial:{partial}")
    print(f"window_size:{window_size}")

    if in_dir.is_file():
        in_files = [in_dir]
    else:
        in_files = [f for f in in_dir.iterdir() if f.is_file()]
    for in_file in in_files:
        if color_type in in_file.stem:
            
            out_file = out_dir / in_file.stem
            out_file.mkdir(parents=True, exist_ok=True)
            in_file = str(in_file)

            in_mask = None
            if in_mask_pat != None:
                in_mask = in_mask_pat.format(image=str(in_file.stem))
            create_segment_job(in_file, in_mask, out_file, color_type, level, partial, window_size)

    return

def create_segment_job(in_file, in_mask, out_file, color_type, level, partial, window_size):
    print("### PARAMETERS ###")
    print(f"in_file:{in_file}")
    print(f"out_file:{out_file}")

    print("# image for MRI stats")
    image_reader = mir.MultiResolutionImageReader()
    image = image_reader.open(in_file)
    dsl = int(image.getLevelDownsample(level))

    if partial:
        print("# perform segmentation")
        result = partial_segmentation(in_file, in_mask, out_file, color_type, window_size, level, dsl)
        in_file=None

    if not partial:
        size = list(image.getLevelDimensions(level))
        in_file = image.getUCharPatch(0, 0, *size, level)

        print("# perform segmentation")
        if color_type == 'PSR':
            result = PSR_segmentaiton.fibrosis_segmentation(in_file)
        
        else:
            sta_result = HNE_segmentation.staetosis_segmentation(in_file, level, dsl)
            inf_result = HNE_segmentation.inflamation_segmentation(in_file, level, dsl)
            result = [sta_result, inf_result]

    print("# write results")
    write_results(in_file, out_file, color_type, result)

    return

def write_results(extracted_image, out_file, color_type, result):

    if color_type == 'PSR':
        fib_result, cpa = result

        print("### RESULT ###")
        print(f"CPA:{round(cpa, 4)}")

        # imageio.imwrite(out_file / ('_result' + '.tif'), fib_result)
        imageio.imwrite(out_file / ('_result' + '.png'), fib_result)

        if extracted_image is not None:
            imageio.imwrite(out_file / ('_original' + '.png'), extracted_image)
            extracted_image[(fib_result == 1).squeeze()] = (0, 0, 255)
            imageio.imwrite(out_file / ('_detection' + '.png'), extracted_image)

        with open(out_file / 'stats.txt', 'w+') as out_file:
            out_file.write(f"CPA:{round(cpa, 4)}")

    else:
        sta_result, inf_result = result
        
        inf_result[inf_result == 2] = 2
        inf_result[inf_result == 1] = 0
        if sta_result is not None:
            sta_result[sta_result >= 1] = 1
            sta_result = sta_result + np.expand_dims(inf_result, axis=-1)
        else:
            sta_result = np.expand_dims(inf_result, axis=-1)
        sta_result[sta_result > 2] = 2

        # imageio.imwrite(out_file / ('_result' + '.tif'), sta_result)
        imageio.imwrite(out_file / ('_result' + '.png'), sta_result.squeeze())
        if extracted_image is not None:
            imageio.imwrite(out_file / ('_original' + '.png'), extracted_image)

            extracted_image[(sta_result == 1).squeeze()] = (255, 0, 0)
            extracted_image[(sta_result == 2).squeeze()] = (0, 255, 0)
            imageio.imwrite(out_file / ('_detection' + '.png'), extracted_image)        

    return


def partial_segmentation(
    in_file,
    in_mask,
    out_file,
    color_type,
    window_size,
    level,
    dsl
    ):

    data_gen = dataGenerators.WSIInferenceSet(in_file, level, patch_size=(window_size, window_size), device='cpu', totensor=False, mask_file=in_mask)
    patchNr = data_gen.getPatchNr()
    size = len(data_gen)

    tmp_out_file = out_file / 'temp'
    tmp_out_file.mkdir(parents=True, exist_ok=True)

    if (size // 10) == 0:
        updateCount = 2
    else: updateCount = (size // 10)
    
    result_arr1 = []
    result_arr2 = []

    mask = None
    for idx in range(len(data_gen)):
        patch = data_gen[idx]
        if len(patch) == 2:
            patch, mask = patch
                
        if mask is not None and np.sum(mask) == 0:
            if color_type == 'PSR':
                result1 = np.zeros((window_size, window_size, 1), dtype=np.uint8)
                result2 = 0
            else:
                result1 = np.zeros((window_size, window_size, 1), dtype=np.uint8)
                result2 = np.zeros((window_size, window_size), dtype=np.uint8)

        elif color_type == 'PSR':
            result1, result2 = PSR_segmentaiton.fibrosis_segmentation(patch)
            result1 = result1.astype(np.uint8)
        
        else:
            result1 = HNE_segmentation.staetosis_segmentation(patch, level, dsl)
            result2 = HNE_segmentation.inflamation_segmentation(patch, level, dsl)
            result1, result2 = result1.astype(np.uint8), result2.astype(np.uint8)
        
        result_arr1.append(result1)
        result_arr2.append(result2)

        if idx % updateCount == 0:
            print(f'at {idx} of {size}')

        if idx % patchNr[0] == 0:
            rowNr = idx // patchNr[0]
            if color_type == 'PSR':
                save_tmp_result(result_arr1, 'f', rowNr, tmp_out_file)
                result_arr1 = []

                file_name = 'running_cpg' + '.npy'
                result_arr2 = sum(result_arr2) / len(result_arr2)

                if (tmp_out_file / file_name).is_file():
                    with open(tmp_out_file / file_name, 'rb') as f:
                        result_arr2 = (result_arr2 + np.load(f))/2

                with open(tmp_out_file / file_name, 'w+') as f:
                    result_arr2 = np.concatenate(result_arr2, axis=2)
                    np.save(np.array(result_arr2), f)
                    result_arr2 = []
            else:
                save_tmp_result(result_arr1, 's', rowNr, tmp_out_file)
                result_arr1 = []

                save_tmp_result(result_arr2, 'i', rowNr, tmp_out_file)
                result_arr2 = []
    
    if len(result_arr1) != 0:
        print(f'resulting patches {len(result_arr1)}, {len(result_arr2)}')
        return 0
    
    if color_type == 'PSR':
        result_arr2 = load_result('f', patchNr, tmp_out_file)
        with open(tmp_out_file / file_name, 'rb') as f:
            result_arr2 = np.load(f)

    else:
        result_arr1 = load_result('s', patchNr, tmp_out_file)
        result_arr2 = load_result('i', patchNr, tmp_out_file)
    
    return result_arr1, result_arr2

def save_tmp_result(result_arr, task, rowNr, tmp_out_file):
    file_name = task + str(rowNr) + '.npy'
    with open(tmp_out_file / file_name, 'w+') as f:
        result_arr = np.concatenate(result_arr, axis=2)
        np.save(f, np.array(result_arr))

def load_result(task, patchNr, tmp_out_file):
    result_arr = []

    for i in range(patchNr[0]):
        print(f'loading file {i}')
        file_name = task + str(i) + '.npy'
        with open(tmp_out_file / file_name, 'rb') as f:
            result_arr.append(np.load(f))

    result_arr = [np.concatenate(result_arr[(i*(patchNr[0])):((i+1)*patchNr[0])], axis=2) for i in range(patchNr[1])]
    result_arr = np.concatenate(result_arr, axis=1)

    return result_arr

# Deprecated
def get_background_thresh(in_img):
    thresh_r = skf.threshold_otsu(in_img[:, :, 0], nbins=100)
    thresh_g = skf.threshold_otsu(in_img[:, :, 1], nbins=100)
    thresh_b = skf.threshold_otsu(in_img[:, :, 2], nbins=100)

    return thresh_r, thresh_g, thresh_b

if __name__ == "__main__":
    main()

