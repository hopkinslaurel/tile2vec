# make_triplets_synthetic.py
# =============================================================================
# Original code by Neal Jean: nealjean/pixel2vec/notebooks/NJ5_naip_sampling*
# Samples triplets (anchor, neighbor, distant) from folder of tif or npy
# images. Anchor and neighbor from same image file, distant from different
# file. Minor extensions/edits by Anshul Samar.
# =============================================================================
# Modificatoin by Laurel Hopkins. This is to be used on clipped data when only
# npy tiles need to be grouped into triplets (don't need to define tile sizes
# and neighborhoods -- already est. with clipped data). Use with triplet csv. 

import numpy as np
import os
import random
from utils import *
import paths
import argparse
import datetime
import pickle
import csv

parser = argparse.ArgumentParser()
parser.add_argument('--ntrain', dest='ntrain', type=int, default=0)
parser.add_argument('--nval', dest='nval', type=int, default=0)
parser.add_argument('--ntest', dest='ntest', type=int, default=0)
parser.add_argument('-debug',  action='store_true')
args = parser.parse_args()
print(args)

def import_triplets():
    triplets = {}
    with open('OR_2011_synthetic_triplets.csv', 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            triplets[row['anchor']] = {'neighbor':int(row['neighbor']), 'distant':int(row['distant'])}
    return triplets


def make_triplets (tile_dir, trip_dir_train, trip_dir_test, trip_dir_val, num_train, num_test, num_val):
    # get number/percent of test, train, val
    # read in csv with neighbor, anchor, distant
    triplets = import_triplets()
    # randomly partition data into train/test/val
    total = num_train + num_test + num_val
    # for img in csv, make triplet -- rename to idx_anchor, idx_neighbor, idx_distnat
    # (save image to associated directory)
    for anchor in triplets:
        print(anchor)
        anchor, neighbor, distant = int(anchor), triplets[anchor]['neighbor'], triplets[anchor]['distant']
        tile_anchor = np.load(os.path.join(tile_dir, 'synthetic_naip_oregon_2011_20m_{}.npy'.format(anchor)))
        tile_neighbor = np.load(os.path.join(tile_dir, 'synthetic_naip_oregon_2011_20m_{}.npy'.format(neighbor)))
        tile_distant = np.load(os.path.join(tile_dir, 'synthetic_naip_oregon_2011_20m_{}.npy'.format(distant)))
        rand = np.random.randint(0,total)
        if rand <= num_train:
            path = trip_dir_train
        elif rand > num_train and rand <= num_train+num_test:
            path = trip_dir_test
        elif rand > num_train+num_test and rand <= total:
            path = trip_dir_val
        np.save(os.path.join(path, '{}anchor.npy'.format(anchor)), tile_anchor)
        np.save(os.path.join(path, '{}neighbor.npy'.format(anchor)), tile_neighbor)
        np.save(os.path.join(path, '{}distant.npy'.format(anchor)), tile_distant)


# Run
if args.debug:
    np.random.seed(1)

print("Generating Triplets")
make_triplets(paths.synthetic_tiles, paths.synthetic_triplets_train, paths.synthetic_triplets_test, 
              paths.synthetic_triplets_val, args.ntrain, args.ntest, args.nval)

