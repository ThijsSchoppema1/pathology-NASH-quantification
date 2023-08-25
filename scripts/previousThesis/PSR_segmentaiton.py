import numpy as np
import skimage.filters as skf
import cv2
from sklearn.cluster import KMeans
from sklearn.utils import shuffle
from scipy import ndimage

def background_filter(in_img):
    mask_shape = list(in_img.shape)
    mask_shape[2] = 1
    mask = np.ones(mask_shape, dtype=int)

    mask[in_img[:, :, 0] < 200] = 0
    mask[in_img[:, :, 2] > 210] = 0

    kernel = np.ones((5,5),np.uint8)
    closing = cv2.morphologyEx(np.float64(mask.squeeze()), cv2.MORPH_CLOSE, kernel)
    mask = cv2.morphologyEx(closing, cv2.MORPH_OPEN, kernel)
    mask = np.expand_dims(mask, axis=2).astype(np.int32)

    return mask

def fibrosis_segmentation(in_img):
    # Params

    # Create mask
    print("# Create mask")
    in_img_shape = in_img.shape
    bg_mask = background_filter(in_img)
    masked_img = in_img * bg_mask

    # KMEANS clustering
    print("# Kmeans clustering")

    w, h, d = tuple(masked_img.shape)
    vec_masked_img = np.reshape(masked_img, (w * h, d))

    image_array_sample = shuffle(vec_masked_img, random_state=0, n_samples=100_000)
    kmeans = KMeans(n_clusters=3, n_init=10, random_state=0).fit(
        image_array_sample
    )
    km_labels = kmeans.predict(vec_masked_img)
    km_centers = kmeans.cluster_centers_

    # Create Fibrosis mask
    print("# Fibrosis mask")
    background_label = np.argmin(np.mean(km_centers, axis=1))
    tissue_label = np.argmax(np.mean(km_centers, axis=1))
    fibrosis_label = 3 - (background_label + tissue_label)

    _, counts = np.unique(km_labels, return_counts=True)
    
    fib_mask_shape = list(in_img_shape)
    fib_mask_shape[2] = 1
    fib_mask = np.zeros(fib_mask_shape, dtype=int)

    km_mask = km_labels.reshape((fib_mask_shape))
    fib_mask[km_mask == fibrosis_label] = 1

    fib_mask[masked_img[:, :, 2] > 175] = 0

    print("# Border check")
    bg_mask = np.array(bg_mask, np.uint8)
    cnts, _ = cv2.findContours(bg_mask.squeeze(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    s = list(fib_mask.shape)
    s[2] = 3
    bimgs = np.zeros(s)
    bimgs = np.array(bimgs, np.uint8)
    cv2.drawContours(bimgs, cnts, -1, (255, 255, 255), 3)
    bimgs = cv2.cvtColor(bimgs, cv2.COLOR_BGR2GRAY)
    bimgs[bimgs!=0]=255
    fib_mask[bimgs == 255] = 2

    # Look for labelled areas which connect with the tissue borders
    labeled_array, num_features = ndimage.label(fib_mask)
    max_vals = ndimage.maximum(fib_mask,labeled_array,range(1,num_features+1))
    m_idx = np.where(max_vals==2)[0] + 1 

    # remove those areas
    max_index = np.ones(num_features + 1, np.uint8)
    max_index[m_idx] = 0
    fib_mask = fib_mask * max_index[labeled_array]

    if fibrosis_label == 3: cpa = 0
    else:
        cpa = np.sum(fib_mask)/(counts[tissue_label] + counts[fibrosis_label])

    return fib_mask, cpa