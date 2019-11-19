# extract_features.py


import sys
import paths 

tile2vec_dir = paths.home_dir  # '/home/asamar/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)

import os
import torch
from torch import optim
from time import time
import random
import numpy as np
from fig_utils import *
from src.datasets import triplet_dataloader
from src.training import train_model, validate_model, prep_triplets
from utils import *
import paths
from tensorboardX import SummaryWriter
import argparse
import pickle

# Parse command line arguments
parser = argparse.ArgumentParser()

# Training
parser.add_argument('-train', action='store_true')
parser.add_argument('--ntrain', dest='ntrain', type=int, default=100000)
parser.add_argument('-lsms_train',  action='store_true')
parser.add_argument('--nlsms_train', dest='nlsms_train', type=int,
                    default=100000)

# Testing
parser.add_argument('-test',  action='store_true')
parser.add_argument('--ntest', dest='ntest', type=int, default=50000)
parser.add_argument('-lsms_val',  action='store_true')
parser.add_argument('--nlsms_val', dest='nlsms_val', type=int, default=50000)
parser.add_argument('-val',  action='store_true')
parser.add_argument('--nval', dest='nval', type=int, default=50000)

# Regression
parser.add_argument('-predict_small', action='store_true')
parser.add_argument('-predict_big', action='store_true')
parser.add_argument('-quantile', action='store_true')
parser.add_argument('--trials', dest="trials", type=int, default=10)

# Model
parser.add_argument('--model', dest="model", default="tilenet")
parser.add_argument('--z_dim', dest="z_dim", type=int, default=512)
parser.add_argument('--model_fn', dest='model_fn')
parser.add_argument('--exp_name', dest='exp_name')
parser.add_argument('--epochs_end', dest="epochs_end", type=int, default=50)
parser.add_argument('--epochs_start', dest="epochs_start", type=int, default=0)
parser.add_argument('-save_models', action='store_true')
parser.add_argument('--gpu', dest="gpu", type=int, default=0)

# Debug
parser.add_argument('-debug', action='store_true')

# Feature extraction
parser.add_argument('-extract_small', action='store_true')

args = parser.parse_args()
print(args)


# Load Model Definition
if args.model == "minires":
    from src.minires import make_tilenet
elif args.model == "miniminires":
    from src.miniminires import make_tilenet
else:
    from src.tilenet import make_tilenet

if not args.save_models:
    print("Not Saving Checkpoints")

# Environment
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ['CUDA_VISIBLE_DEVICES'] = str(args.gpu)
cuda = torch.cuda.is_available()
if args.debug:
    torch.manual_seed(1)
    if cuda:
        # Not tested if this works/see pytorch thread it may not
        print("Cuda available")
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True


# Data Parameters
img_type = 'rgb'
bands = 3
augment = True
batch_size = 50
shuffle = True
num_workers = 4
test_imgs = 1000

# Training Parameters
in_channels = bands
TileNet = make_tilenet(in_channels=in_channels, z_dim=args.z_dim)
if cuda: TileNet.cuda()


# Load saved model
if args.model_fn:
    TileNet.load_state_dict(torch.load(args.model_fn))
    print('Saved TileNet loaded')


if args.extract_small:
    # Small Image Features
    print("Extracting Features")
    img_names = [paths.lsms_images_small + 'naip_oregon_2011_cluster_' \
                 + str(i) + '.tif' for i in range(test_imgs)]
    #print(*img_names)
    X = get_small_features(img_names, TileNet, args.z_dim, cuda, bands,
                           patch_size=50, patch_per_img=10, save=True,
                           verbose=True, npy=False, quantile=args.quantile)
    print(X.shape)
    
    np.save(paths.ebird_features + 'cluster_conv_features_' + args.exp_name +\
            '.npy', X)
    np.savetxt(paths.ebird_features + 'cluster_conv_features_' + args.exp_name +\
            '.csv', X, delimiter=",")    

    print('Finished')
