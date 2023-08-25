import numpy as np
import cv2
from scipy import ndimage
from skimage.transform import hough_circle
from skimage.transform import hough_circle_peaks
from skimage.draw import disk

def background_filter(in_img, dsl):
    mask_shape = list(in_img.shape)
    mask_shape[2] = 1
    mask = np.zeros(mask_shape, dtype=int)

    mask[in_img[:, :, 0] < 225] = 1
    mask[in_img[:, :, 1] < 200] = 1
    mask[in_img[:, :, 2] < 215] = 1

    kernel = np.ones((int(25 // dsl),int(25 // dsl)),np.uint8)
    mask = cv2.morphologyEx(np.float64(mask.squeeze()), cv2.MORPH_CLOSE, kernel)
    mask = np.expand_dims(mask, axis=2).astype(np.int32)

    return mask

def staetosis_segmentation(in_img, level, dsl):
    rs = (int(50 // dsl), int(100 // dsl))

    # Create mask
    mask = background_filter(in_img, dsl)

    # Detect edges
    fil = [[-1,-1,-1],
          [-1, 8,-1],
         [-1,-1,-1]]
    t = ndimage.convolve(mask.squeeze(),fil, mode='constant')
    t[t<0] = 0
    t[t>0] = 1

    # Hough transformation
    hough_radii = np.arange(rs[0], rs[1], 1)
    hough_res = hough_circle(t, hough_radii)

    _, cx, cy, radii = hough_circle_peaks(hough_res, hough_radii, min_xdistance=100 // dsl, min_ydistance=100 // dsl, num_peaks=2 // dsl)
    
    # map detected circles to the original mask
    mask = np.where(mask == 1, 0, 1)
    for center_y, center_x, radius in zip(cy, cx, radii):
        circy, circx = disk((center_y, center_x), radius,
                                shape=mask.shape)
        mask[circy, circx] = np.where(mask[circy, circx] == 1, 2, mask[circy, circx])
    
    # expand the label areas
    labeled_array, num_features = ndimage.label(mask)
    for i in range(num_features):
        try:
            max_val = np.amax(mask[labeled_array == i])
        except:
            max_val = 0
        mask[labeled_array == i] = max_val

    mask = np.where(mask >= 2, 1, 0)
        
    return mask

def inflamation_segmentation(in_img, level, dsl):
    kernel_sizes = {0:11, 1:7, 2:3, 3:2}
    ks=3
    if level in kernel_sizes:
        ks = kernel_sizes[level]

    # Mask background, count tissue pixels
    mask = background_filter(in_img, dsl)

    mask = np.array(mask, np.uint8)
    cnts = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]
    
    total_pixels = mask.sum()
    n_parts = 0
    for c in cnts:
        pixels = cv2.contourArea(c)
        if pixels / total_pixels > (0.25):
            n_parts += 1
    
    if n_parts == 0: n_parts = 1

    # only select nuclei
    mask[(in_img[:, :, 0] > 190) & (in_img[:, :, 1] > 140) & (in_img[:, :, 2] > 190)] = 0
    kernel = np.ones((ks,ks),np.uint8)
    mask = cv2.dilate(np.float64(mask.squeeze()),kernel,iterations = 1)
    mask = np.array(mask, np.uint8)

    # Find countours and compare to tissue area
    cnts = cv2.findContours(mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] if len(cnts) == 2 else cnts[1]

    counter = 0
    pixels = 0
    for c in cnts:
        pixels = cv2.contourArea(c)
        if pixels / total_pixels > (0.015 / n_parts) / dsl:
            cv2.fillPoly(mask, [c], 2)
            counter += 1

    return mask

def segment_both(patch, level=1, dsl=2.0):

    sta_result = staetosis_segmentation(patch, level, dsl)
    inf_result = inflamation_segmentation(patch, level, dsl)

    inf_result, sta_result = inf_result.astype(np.uint8), sta_result.astype(np.uint8)

    inf_result[inf_result == 2] = 2
    inf_result[inf_result == 1] = 0
    if sta_result is not None:
        sta_result[sta_result >= 1] = 1
        sta_result = sta_result + np.expand_dims(inf_result, axis=-1)
    else:
        sta_result = np.expand_dims(inf_result, axis=-1)
    sta_result[sta_result > 2] = 2

    return sta_result