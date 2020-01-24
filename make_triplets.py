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
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import colors
import argparse
import datetime
import pickle
import string
import subprocess

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('--ntrain', dest='ntrain', type=int, default=100000)
parser.add_argument('-val', action='store_true')
parser.add_argument('--nval', dest='nval', type=int, default=10000)
parser.add_argument('-test',  action='store_true')
parser.add_argument('--ntest', dest='ntest', type=int, default=10000)
parser.add_argument('-lsms_train',  action='store_true')
parser.add_argument('--nlsms_train', dest='nlsms_train', type=int, default=10000)
parser.add_argument('-lsms_val',  action='store_true')
parser.add_argument('--nlsms_val', dest='nlsms_val', type=int, default=10000)
parser.add_argument('--nghbr', dest='nghbr', type=int, default=50)
parser.add_argument('-debug',  action='store_true')
parser.add_argument('-color_map',  action='store_true')
parser.add_argument('--bands', dest='bands', type=int, default=3)
parser.add_argument('-smart', action='store_true')
args = parser.parse_args()
print(args)

# Assumes 8x8 grid of clusters (of 16x16 .npy files, row major order)
def get_coord (filename):
    number = int(filename[0:len(filename) - len('.npy')])
    cluster = number // (100*100) #16*16
    cluster_row = cluster // 20  #8
    cluster_col = cluster % 20  #8
    image_idx = number % (100*100)
    image_row = image_idx // 100
    image_col = image_idx % 100
    return cluster_row*100 + image_row, cluster_col*100 + image_col

def train_map(color_map):
    count = 0
    for filename in sorted(os.listdir(paths.train_images)):
        if filename.endswith('.npy'):
            row, col = get_coord(filename)
            color_map[row][col] = 1
            color_map[row][col] = 1
            color_map[row][col] = 1
            count += 1
    print(str("Files: ") + str(count))

def test_map(color_map):
    count = 0
    for filename in sorted(os.listdir(paths.test_images)):
        if filename.endswith('.npy'):
            row, col = get_coord(filename)
            color_map[row][col] = 3
            color_map[row][col] = 3
            color_map[row][col] = 3
            count += 1
    print(str("Files: ") + str(count))

def get_triplets (data_dir, tile_dir, num_triplets, bands=7, tile_size=50,
                       neighborhood=125, npy=True, map_type="", smart):
    size_even = (tile_size % 2 == 0)
    tile_radius = tile_size // 2
    alph = list(string.ascii_lowercase[0:26])
    tiles = np.zeros((num_triplets, 3, 2), dtype=np.int16) #TODO: does this need to change for several neighbor and distant potential?
    grid = sorted(os.listdir(data_dir))
    seen = set()
    for i in range(0, num_triplets):
        near_img, far_img = np.random.choice(grid, 2, replace=False)
        if map_type in ["train","val","test"] and args.color_map is not None:
            row, col = get_coord(near_img)
            if (row,col) not in seen:
                seen.add((row,col))
            if map_type == "train":
                color_map[row][col] = 2
            elif map_type == "test":
                color_map[row][col] = 4
            row, col = get_coord(far_img)
            if (row,col) not in seen:
                seen.add((row,col))
            if map_type == "train":
                color_map[row][col] = 2
            elif map_type == "test":
                color_map[row][col] = 4
        print(str(i) + ": " + str(near_img) + "," + str(far_img))
        near_img = load_landsat(data_dir + near_img, bands, bands_only=True, is_npy=npy)
        far_img = load_landsat(data_dir + far_img, bands, bands_only=True, is_npy=npy)
        img_shape = near_img.shape
        num_triplet_options = 1
        xa, ya = sample_tile(img_shape, tile_radius) # get single anchor
        tile_anchor = extract_patch(near_img, xa, ya, tile_radius, args.bands)
        if size_even:
            tile_anchor = tile_anchor[:-1,:-1]
        np.save(os.path.join(tile_dir, '{}anchor.npy'.format(i)), tile_anchor)
        if smart:
            num_triplet_options = 20 #TODO: could tune this number 
        xn, yn = np.zeros(num_triplet_options), np.zeros(num_triplet_options)
        xd, yd = np.zeros(num_triplet_options), np.zeros(num_triplet_options)
        for j in range(0, num_triplet_options):
            xn[j], yn[j] = sample_neighbor(img_shape, xa, ya, neighborhood, tile_radius)
            img_shape = far_img.shape
            xd[j], yd[j] = sample_tile(img_shape, tile_radius)
            tile_anchor = extract_patch(near_img, xa, ya, tile_radius, args.bands)
            tile_neighbor = extract_patch(near_img, xn, yn, tile_radius, args.bands)
            tile_distant = extract_patch(far_img, xd, yd, tile_radius, args.bands)
            if size_even:
                tile_neighbor = tile_neighbor[:-1,:-1]
                tile_distant = tile_distant[:-1,:-1]
            np.save(os.path.join(tile_dir, '{}_{}_neighbor.npy'.format(i,alph[j])), tile_neighbor)
            np.save(os.path.join(tile_dir, '{}_{}_distant.npy'.format(i,alph[j])), tile_distant)
                
            #plt.imsave('{}anchor.jpg'.format(i), tile_anchor)
            #plt.imsave('{}neighbor.jpg'.format(i), tile_neighbor)
            #plt.imsave('{}distant.jpg'.format(i), tile_distant)

            # TODO: Call R script here
            subprocess.call('/home/hopkilau/_tile2vec/test.R')
                    
            #TODO: Read in something saved either by R OR remaining files, determine which character, convert to value and then save 
            ind_neighbor = 0#replace
            ind_distant = 0#replace
            
        tiles[i,0,:] = xa, ya
        tiles[i,1,:] = xn[ind_neighbor], yn[ind_neighbor]
        tiles[i,2,:] = xd[ind_distant], yd[ind_distant]
    print(len(seen))
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

# Color Map (thanks stackoverflow user @umutto)

color_map = np.zeros((8*16,8*16))
cmap = colors.ListedColormap(['xkcd:black', 'xkcd:light blue',
                              'xkcd:bright blue',
                              'xkcd:spring green', 'xkcd:green'])
bounds = [0,1,2,3,4,5]
norm = colors.BoundaryNorm(bounds, cmap.N)
fig, ax = plt.subplots(figsize=(8,8))
bands = 3

if args.train:
    print("Generating Train Set")
    #train_map(color_map)  
    tiles_train = get_triplets(paths.train_images, paths.train_tiles,
                               args.ntrain, bands, tile_size = 200,  # tile_size = 200(*10m/pixel) = 2km
                               neighborhood = args.nghbr,
                               npy = True, map_type="train", args.smart)    

if args.val:
    print("Generating Val Set")
    tiles_val = get_triplets(paths.train_images, paths.val_tiles, args.nval,
                             bands, tile_size = 200, neighborhood = args.nghbr,
                             npy = True, map_type="val", args.smart)
    
if args.test:
    print("Generating Test Set")
    #test_map(color_map)
    tiles_test = get_triplets(paths.test_images, paths.test_tiles, 
                              args.ntest, bands, tile_size = 200,
                              neighborhood = args.nghbr, npy = True,
                              map_type = "test", args.smart)                                

if args.lsms_train:
    print("Generating LSMS Train Set")
    tiles_lsms = get_triplets(paths.lsms_images_big, paths.lsms_train_tiles, 
                              args.nlsms_train, bands, tile_size = 200,
                              neighborhood = args.nghbr, npy= False,
                              map_type = "lsms", args.smart)

if args.lsms_val:
    print("Generating LSMS Val Set")
    tiles_lsms = get_triplets(paths.lsms_images_big, paths.lsms_val_tiles, 
                              args.nlsms_val, bands, tile_size = 200,
                              neighborhood = args.nghbr, npy= False,
                              map_type = "lsms", args.smart)

    
ax.imshow(color_map, cmap=cmap, norm=norm)
now = datetime.datetime.now()
ax.axis('off')
#ax.yaxis.grid(which="major", color='black', linestyle='-', linewidth=1)
#ax.xaxis.grid(which="major", color='black', linestyle='-', linewidth=1)
#ax.set_xticks(np.arange(0, 8*16, 16));
#ax.set_yticks(np.arange(0, 8*16, 16));
plt.savefig("color_map_" + now.isoformat() + ".png")
with open("color_map_" + now.isoformat() + ".p","wb") as f:
    pickle.dump(color_map,f)
