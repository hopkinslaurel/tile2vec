import numpy as np
import os
import gdal
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from time import time
from src.data_utils import clip_and_scale_image
import torch
from torch.autograd import Variable

def sample_patch(img_shape, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    xa = np.random.randint(0, w) + patch_radius
    ya = np.random.randint(0, h) + patch_radius
    return xa, ya

def load_landsat_npy(img_fn, bands, bands_only=False):
    img = np.load(img_fn)
    if bands_only: img = img[:,:,:bands]
    return img

def load_landsat(img_fn, bands, bands_only=False, is_npy=True):
    """
    Loads Landsat image with gdal, returns image as array.
    Move bands (i.e. r,g,b,etc) to last dimension. 
    """
    if is_npy:
        return load_landsat_npy(img_fn, bands, bands_only)
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.uint8)
    del obj # close GDAL dataset
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:bands]
    return img

def extract_patch(img_padded, x0, y0, patch_radius):
    """
    Extracts a patch from a (padded) image given the row and column of
    the center pixel and the patch size. E.g., if the patch
    size is 15 pixels per side, then the patch radius should be 7.
    """
    w_padded, h_padded, c = img_padded.shape
    row_min = x0 - patch_radius
    row_max = x0 + patch_radius
    col_min = y0 - patch_radius
    col_max = y0 + patch_radius
    assert row_min >= 0, 'Row min: {}'.format(row_min)
    assert row_max <= w_padded, 'Row max: {}'.format(row_max)
    assert col_min >= 0, 'Col min: {}'.format(col_min)
    assert col_max <= h_padded, 'Col max: {}'.format(col_max)
    patch = img_padded[row_min:row_max+1, col_min:col_max+1, :]
    return patch

def get_test_features (img_name, model, z_dim, cuda, bands=7, patch_size=50,
                       patch_per_img=10, save=True, verbose=False, npy=True):
    if verbose:
        print("Getting features for: " + img_name)
    patch_radius = patch_size // 2   
    img = load_landsat(img_name, bands, bands_only=True, is_npy=npy)
    img_shape = img.shape
    output = np.zeros((1,z_dim))
    for i in range(patch_per_img):
        xa, ya = sample_patch(img_shape, patch_radius)
        patch = extract_patch(img, xa, ya, patch_radius)
        patch = np.moveaxis(patch, -1, 0)
        patch = np.expand_dims(patch, axis=0)
        patch = clip_and_scale_image(patch, 'landsat')
        # Embed tile
        patch = torch.from_numpy(patch).float()
        patch = Variable(patch)
        if cuda: patch = patch.cuda()
        z = model.encode(patch)
        if cuda: z = z.cpu()
        z = z.data.numpy()
        output += z
    output = output / patch_per_img
    return output
