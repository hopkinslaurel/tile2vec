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
from collections import defaultdict

parser = argparse.ArgumentParser()
parser.add_argument('--ntrain', dest='ntrain', type=int, default=0)
parser.add_argument('--nval', dest='nval', type=int, default=0)
parser.add_argument('--ntest', dest='ntest', type=int, default=0)
parser.add_argument('--file', dest='file', type=str)
parser.add_argument('-manual_split', action='store_true')
parser.add_argument('-debug',  action='store_true')
parser.add_argument('-duplicates', action='store_true')
args = parser.parse_args()
print(args)

def import_triplets():
    triplets = {}
    print('Extracting triplets from {}'.format(args.file))
    with open(args.file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            triplets[row['anchor']] = {'neighbor':int(row['neighbor']), 'distant':int(row['distant'])}
    return triplets


def import_triplets_duplicate_anchor():
    triplets = defaultdict(list)
    print('Extracting triplets from {}'.format(args.file))
    with open(args.file, 'r') as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader: 
            triplets[row['anchor']].append((int(row['neighbor']),int(row['distant'])))
    return triplets


def import_split():
    train, test, val = [], [], []
    with open('OR_2011_train_IDs_synthetic_50k.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            train.append(row[1])
    with open('OR_2011_test_IDs_synthetic_50k.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            test.append(row[1])
    with open('OR_2011_val_IDs_synthetic_50k.csv', 'r') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)
        for row in reader:
            val.append(row[1])
    return (train, test, val)


def make_triplets(tile_dir, trip_dir_train, trip_dir_test, trip_dir_val, num_train, num_test, num_val):
    triplets = import_triplets()
    print("Making " + str(len(triplets)) + " triplets")
    if args.manual_split:
        train, test, val = import_split()
        print("Imported {} train IDS".format(len(train)))
        print("Imported {} test IDS".format(len(test)))
        print("Imported {} val IDS".format(len(val)))
    for anchor in triplets:
        print(anchor)
        anchor, neighbor, distant = anchor, triplets[anchor]['neighbor'], triplets[anchor]['distant']
        tile_anchor = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(int(anchor))))
        tile_neighbor = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(neighbor)))
        tile_distant = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(distant)))
        if args.manual_split:
            if anchor in train:
                path = trip_dir_train
            elif anchor in test:
                path = trip_dir_test
            elif anchor in val:
                path = trip_dir_val            
        else:
            total = num_train + num_test + num_val
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
       
    
def make_triplets_duplicate_anchor(tile_dir, trip_dir_train, trip_dir_test, trip_dir_val, num_train, num_test, num_val):
    triplets = import_triplets_duplicate_anchor()
    print("Making " + str(len(triplets)) + " triplets")
    if args.manual_split:
        train, test, val = import_split()
        print("Imported {} train IDS".format(len(train)))
        print("Imported {} test IDS".format(len(test)))
        print("Imported {} val IDS".format(len(val)))
        #alph = ['a','b','c','d','e','f','g','h','i','j']
        idx = 0
    for anchor in triplets:
        print(anchor)
        for neighbor, distant in triplets[anchor]:
            tile_anchor = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(int(anchor))))
            tile_neighbor = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(neighbor)))
            tile_distant = np.load(os.path.join(tile_dir, 'synthetic_naip_10m_15k_oregon_{}.npy'.format(distant)))
            if args.manual_split:
                if anchor in train:
                    path = trip_dir_train
                elif anchor in test:
                    path = trip_dir_test
                elif anchor in val:
                    path = trip_dir_val
            else:
                total = num_train + num_test + num_val
                rand = np.random.randint(0,total)
                if rand <= num_train:
                    path = trip_dir_train
                elif rand > num_train and rand <= num_train+num_test:
                    path = trip_dir_test
                elif rand > num_train+num_test and rand <= total:
                    path = trip_dir_val
            np.save(os.path.join(path, '{}anchor.npy'.format(idx)), tile_anchor)
            np.save(os.path.join(path, '{}neighbor.npy'.format(idx)), tile_neighbor)
            np.save(os.path.join(path, '{}distant.npy'.format(idx)), tile_distant)
            idx+=1
   

# Run
if args.debug:
    np.random.seed(1)

print("Generating Triplets")
if(args.duplicates):
    make_triplets_duplicate_anchor(paths.synthetic_tiles, paths.synthetic_triplets_train_duplicates, 
            paths.synthetic_triplets_test_duplicates, paths.synthetic_triplets_val_duplicates, 
            args.ntrain, args.ntest, args.nval)
else:
     make_triplets(paths.synthetic_tiles, paths.synthetic_triplets_train, paths.synthetic_triplets_test,
              paths.synthetic_triplets_val, args.ntrain, args.ntest, args.nval)
