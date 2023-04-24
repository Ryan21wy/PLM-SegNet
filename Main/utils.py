"""
Adapted from nnU-Net
https://github.com/MIC-DKFZ/nnUNet/blob/master/nnunetv2/inference/sliding_window_prediction.py
"""


import numpy as np
from scipy.ndimage import gaussian_filter


def compute_gaussian(tile_size, sigma_scale=1. / 4, dtype=np.float16):
    tmp = np.zeros(tile_size)
    center_coords = [i // 2 for i in tile_size]
    sigmas = [i * sigma_scale for i in tile_size]
    tmp[tuple(center_coords)] = 1
    gaussian_importance_map = gaussian_filter(tmp, sigmas, 0, mode='constant', cval=0)
    gaussian_importance_map = gaussian_importance_map / np.max(gaussian_importance_map) * 1
    gaussian_importance_map = gaussian_importance_map.astype(dtype)

    # gaussian_importance_map cannot be 0, otherwise we may end up with nans!
    gaussian_importance_map[gaussian_importance_map == 0] = np.min(
        gaussian_importance_map[gaussian_importance_map != 0])

    return gaussian_importance_map
