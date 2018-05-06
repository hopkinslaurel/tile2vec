# Based on Neal Jean/Sherrie's code

import numpy as np
import os
import gdal
import random
import scipy.stats as stats
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from time import time
from utils import *
import torch
from time import time
from torch.autograd import Variable
from src.tilenet import make_tilenet
from src.resnet import ResNet18
from src.data_utils import clip_and_scale_image

# Setting up model
in_channels = 5
z_dim = 512
cuda = torch.cuda.is_available()
tilenet = make_tilenet(in_channels = in_channels, z_dim=z_dim)
if cuda: tilenet.cuda()

# Load parameters
model_fn = '/home/asamar/tile2vec/models/5_bands/TileNet.ckpt'
tilenet.load_state_dict(torch.load(model_fn))
tilenet.eval()

test_imgs = 642
patches_per_img = 10
X = np.zeros((test_imgs, z_dim))
bands = 5

for i in range(test_imgs):
    img_name = '/home/asamar/tile2vec/data/uganda_landsat_test/landsat7_uganda_3yr_cluster_' + str(i) + '.tif'
    X[i] = get_test_features (img_name, tilenet, z_dim, cuda, bands, patch_size=50, patch_per_img=10, save=True, verbose=True)

np.save('/home/asamar/tile2vec/data/uganda_lsms/cluster_conv_features.npy', X)
