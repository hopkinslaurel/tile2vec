# make_triplets.py
# =============================================================================
# Original code by Neal Jean: nealjean/pixel2vec/notebooks/NJ5_naip_sampling*
# Samples triplets (anchor, neighbor, distant) from folder of tif or npy
# images. Anchor and neighbor from same image file, distant from different
# file. Minor extensions/edits by Anshul Samar. 

import numpy as np
import os
import random
from utils import *
import paths

def get_triplets (data_dir, tile_dir, num_triplets, bands=7, tile_size=50,
                       neighborhood=125, npy=True):
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2
    tiles = np.zeros((num_triplets, 3, 2), dtype=np.int16)
    grid = sorted(os.listdir(data_dir))
    for i in range(0, num_triplets):
        near_img, far_img = np.random.choice(grid, 2, replace=False)
        print(str(i) + ": " + str(near_img) + "," + str(far_img))
        near_img = load_landsat(data_dir + near_img, bands, bands_only=True, is_npy=npy)
        far_img = load_landsat(data_dir + far_img, bands, bands_only=True, is_npy=npy)
        img_shape = near_img.shape
        xa, ya = sample_tile(img_shape, tile_radius)
        xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)
        img_shape = far_img.shape
        xd, yd = sample_tile(img_shape, tile_radius)
        print("Anchor:" + str(xa) + "," + str(ya) + "Neighbor:" + str(xn)
              + "," + str(yn) + " Distant:" + str(xd) + "," + str(yd))

        tile_anchor = extract_patch(near_img, xa, ya, tile_radius)
        tile_neighbor = extract_patch(near_img, xn, yn, tile_radius)
        tile_distant = extract_patch(far_img, xd, yd, tile_radius)
        if size_even:
            tile_anchor = tile_anchor[:-1,:-1]
            tile_neighbor = tile_neighbor[:-1,:-1]
            tile_distant = tile_distant[:-1,:-1]
        np.save(os.path.join(tile_dir, '{}anchor.npy'.format(i)), tile_anchor)
        np.save(os.path.join(tile_dir, '{}neighbor.npy'.format(i)), tile_neighbor)
        np.save(os.path.join(tile_dir, '{}distant.npy'.format(i)), tile_distant)
                
        tiles[i,0,:] = xa, ya
        tiles[i,1,:] = xn, yn
        tiles[i,2,:] = xd, yd

    return tiles

def sample_tile(img_shape, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xa = np.random.randint(0, w) + tile_radius
    ya = np.random.randint(0, h) + tile_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * tile_radius
    h = h_padded - 2 * tile_radius
    
    xn = np.random.randint(max(xa-neighborhood, tile_radius),
                           min(xa+neighborhood, w+tile_radius))
    yn = np.random.randint(max(ya-neighborhood, tile_radius),
                           min(ya+neighborhood, h+tile_radius))
    return xn, yn

# Run
np.random.seed(1)
bands = 11
print("Generating Train Set")
tiles_train = get_triplets(paths.train_images, paths.train_tiles, 
                             100, bands, tile_size =50,
                             neighborhood=50, npy=True)

print("Generating Test Set")
tiles_train = get_triplets(paths.test_images, paths.test_tiles, 
                             100, bands, tile_size =50,
                             neighborhood=50, npy=True)

print("Generating LSMS Set")
tiles_train = get_triplets(paths.lsms_images, paths.lsms_tiles, 
                             100, bands, tile_size =50,
                             neighborhood=50, npy=False)



                    
