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
import matplotlib.pyplot as plt
import colorsys
import argparse

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('--ntrain', dest='ntrain', type=int, default=100000)
parser.add_argument('-val', action='store_true')
parser.add_argument('--nval', dest='nval', type=int, default=10000)
parser.add_argument('-test',  action='store_true')
parser.add_argument('--ntest', dest='ntest', type=int, default=10000)
parser.add_argument('-lsms',  action='store_true')
parser.add_argument('--nlsms', dest='nlsms', type=int, default=10000)
parser.add_argument('--nghbr', dest='nghbr', type=int, default=50)
parser.add_argument('-debug',  action='store_true')
args = parser.parse_args()
print(args)

# Assumes 8x8 grid of clusters (of 16x16 .npy files, row major order)
def get_coord (filename):
    number = int(filename[0:len(filename) - len('.npy')])
    cluster = number // (16*16)
    cluster_row = cluster // 8
    cluster_col = cluster % 8
    image_idx = number % (16*16)
    image_row = image_idx // 16
    image_col = image_idx % 16
    return cluster_row*16 + image_row, cluster_col*16 + image_col

def train_map(color_map):
    for filename in sorted(os.listdir(paths.train_images)):
        if filename.endswith('.npy'):
            row, col = get_coord(filename)
            color_map[row][col][0] = 10
            color_map[row][col][1] = 10
            color_map[row][col][2] = 10

def test_map(color_map):
    for filename in sorted(os.listdir(paths.test_images)):
        if filename.endswith('.npy'):
            row, col = get_coord(filename)
            color_map[row][col][0] = 0
            color_map[row][col][1] = 0
            color_map[row][col][2] = 0

def update_color_map(row, col, seen_train, color_map):
    if (row,col) not in seen_train:
        color_map[row][col][0] = .6
        color_map[row][col][1] = .6
        color_map[row][col][2] = .6
        seen_train.add((row,col))
    else:
        # h, l, s = colorsys.rgb_to_hls(color_map[row][col][0],
        # color_map[row][col][1],
        # color_map[row][col][2])
        # r, g, b = colorsys.hls_to_rgb(h, l + .1, s)
        color_map[row][col][0] += .05
        color_map[row][col][1] += .05
        color_map[row][col][2] += .05

        
def get_triplets (data_dir, tile_dir, num_triplets, bands=7, tile_size=50,
                       neighborhood=125, npy=True, color_map=None, channel=None):
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2
    tiles = np.zeros((num_triplets, 3, 2), dtype=np.int16)
    grid = sorted(os.listdir(data_dir))
    seen_train = set()
    for i in range(0, num_triplets):
        near_img, far_img = np.random.choice(grid, 2, replace=False)
        if color_map is not None:
            row, col = get_coord(near_img)
            update_color_map(row, col, seen_train, color_map)
            row, col = get_coord(far_img)
            update_color_map(row, col, seen_train, color_map)
        print(str(i) + ": " + str(near_img) + "," + str(far_img))
        near_img = load_landsat(data_dir + near_img, bands, bands_only=True, is_npy=npy)
        far_img = load_landsat(data_dir + far_img, bands, bands_only=True, is_npy=npy)
        img_shape = near_img.shape
        xa, ya = sample_tile(img_shape, tile_radius)
        xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)
        img_shape = far_img.shape
        xd, yd = sample_tile(img_shape, tile_radius)
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
    print(len(seen_train))
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
if args.debug:
    np.random.seed(1)
    
color_map = np.zeros((8*16,8*16,3))
test_map(color_map)

bands = 11

if args.train:
    print("Generating Train Set")
    tiles_train = get_triplets(paths.train_images, paths.train_tiles,
                               args.ntrain, bands, tile_size = 50,
                               neighborhood = args.nghbr,
                               npy = True, color_map=color_map)
print(np.amax(color_map))
plt.imsave('color_map.jpg', color_map[0:100,0:100,:])

if args.val:
    print("Generating Val Set")
    tiles_val = get_triplets(paths.train_images, paths.val_tiles, args.nval,
                               bands, tile_size = 50, neighborhood = args.nghbr,
                               npy = True, color_map=color_map)

    
if args.test:
    print("Generating Test Set")
    tiles_test = get_triplets(paths.test_images, paths.test_tiles, 
                              args.ntest, bands, tile_size = 50,
                              neighborhood = args.nghbr, npy = True,
                              color_map=None)

if args.lsms:
    print("Generating LSMS Set")
    tiles_lsms = get_triplets(paths.lsms_images, paths.lsms_tiles, 
                              args.nlsms, bands, tile_size =50,
                              neighborhood = args.nghbr, npy=False,
                              color_map=None)



                    
