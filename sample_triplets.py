# Minor edits to Neal Jean's pixel2vec:
# nealjean/pixel2vec/notebooks/NJ5_naip_sampling*

import numpy as np
import os
import gdal
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from time import time

def load_landsat(img_fn, bands_only=False):
    """
    Loads Landsat image with gdal, returns image as array.
    Move bands (i.e. r,g,b,etc) to last dimension. 
    """
    obj = gdal.Open(img_fn)
    img = obj.ReadAsArray().astype(np.uint8)
    del obj # close GDAL dataset
    img = np.moveaxis(img, 0, -1)
    if bands_only: img = img[:,:,:3]
    return img

def get_triplet_imgs(imgs_dir, n_triplets=1000):
    """
    Create n_triplet random pairs of image filenames.
    Anchor/neighbor patches will come from first file, 
    distant patch from second file.
    imgs_dir: folder of .tif landsat image files
    """
    img_names = []
    for filename in os.listdir(imgs_dir):
        if filename.endswith('.tif'):
            img_names.append(filename)
    img_triplets = list(map(lambda _: random.choice(img_names),
                            range(2 * n_triplets)))
    img_triplets = np.array(img_triplets)
    return img_triplets.reshape((-1, 2))


def get_patch_triplets(patch_dir, img_dir, img_triplets, patch_size=50,
                       neighborhood=125, save=True, verbose=False):
    """
    For each unique image, extracts any anchor
    /neighbor/distant patches as specified by
    img_triplets. 
    Returns "patches" - n_tripletsx3x2 array of x,y
    coordinates of the triplets.
    Saves respective images to patch_dir.
    Assumes input in form (w,h,c)
    """
    t0 = time()
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    size_even = (patch_size % 2 == 0)
    patch_radius = patch_size // 2
    
    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    total_imgs = len(unique_imgs)
    patches = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    save_count = 0
    for img_count, img_name in enumerate(unique_imgs):
        #print('\nStarting image {}/{}: {:0.3f}s\n'.format(img_count+1, total_imgs, time()-t0))
        print("Sampling image {}".format(img_name))
        img = load_landsat(img_dir+img_name, bands_only=True)
        img_padded = np.pad(img, pad_width=[(patch_radius, patch_radius),
                                            (patch_radius, patch_radius), (0,0)],
                            mode='reflect')
        img_shape = img_padded.shape
        
        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                xa, ya = sample_anchor(img_shape, patch_radius)
                xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, patch_radius)
                
                if verbose:
                    print("    Saving anchor and neighbor patch #{}".format(idx))
                    print("    Anchor patch center:{}".format((xa, ya)))
                    print("    Neighbor patch center:{}".format((xn, yn)))
                if save:
                    patch_anchor = extract_patch(img_padded, xa, ya, patch_radius)
                    patch_neighbor = extract_patch(img_padded, xn, yn, patch_radius)
                    if size_even:
                        patch_anchor = patch_anchor[:-1,:-1]
                        patch_neighbor = patch_neighbor[:-1,:-1]
                    np.save(os.path.join(patch_dir, '{}patch.npy'.format(idx)), patch_anchor)
                    np.save(os.path.join(patch_dir, '{}neighbor.npy'.format(idx)), patch_neighbor)
                    save_count += 2
                
                patches[idx,0,:] = xa, ya
                patches[idx,1,:] = xn, yn
                
                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    xd, yd = sample_distant_same(img_shape, xa, ya, neighborhood, patch_radius)
                    if verbose:
                        print("    Saving distant patch #{}".format(idx))
                        print("    Distant patch center:{}".format((xd, yd)))
                    if save:
                        patch_distant = extract_patch(img_padded, xd, yd, patch_radius)
                        if size_even:
                            patch_distant = patch_distant[:-1,:-1]
                        np.save(os.path.join(patch_dir, '{}distant.npy'.format(idx)), patch_distant)
                        save_count += 1
                    patches[idx,2,:] = xd, yd
            
            elif row[1] == img_name: 
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, patch_radius)
                if verbose:
                        print("    Saving distant patch #{}".format(idx))
                        print("    Distant patch center:{}".format((xd, yd)))
                if save:
                    patch_distant = extract_patch(img_padded, xd, yd, patch_radius)
                    if size_even:
                        patch_distant = patch_distant[:-1,:-1]
                    np.save(os.path.join(patch_dir, '{}distant.npy'.format(idx)), patch_distant)
                    save_count += 1
                patches[idx,2,:] = xd, yd
        #print('\nFinished image {}/{}: {} patches saved\n'.format(img_count+1, total_imgs, save_count))
            
    return patches

def sample_anchor(img_shape, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xa = np.random.randint(0, w) + patch_radius
    ya = np.random.randint(0, h) + patch_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xn = np.random.randint(max(xa-neighborhood, patch_radius),
                           min(xa+neighborhood, w+patch_radius))
    yn = np.random.randint(max(ya-neighborhood, patch_radius),
                           min(ya+neighborhood, h+patch_radius))
    return xn, yn

def sample_distant_same(img_shape, xa, ya, neighborhood, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xd, yd = xa, ya
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + patch_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + patch_radius
    return xd, yd

def sample_distant_diff(img_shape, patch_radius):
    return sample_anchor(img_shape, patch_radius)

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

# Test:
data_dir = '/home/asamar/tile2vec/data/uganda_landsat/'
load_landsat(data_dir + 'landsat7_uganda_3yr_row_0_column_1.tif', True)
img_triplets = get_triplet_imgs(data_dir,2)
print(img_triplets)
patches = get_patch_triplets('/home/asamar/tile2vec/data/uganda_patches',
                             data_dir, img_triplets, patch_size =50,
                             neighborhood=125, save=True,verbose=True)



                    
