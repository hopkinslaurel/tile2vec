# Based on Neal Jean/Sherrie

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

def get_test_features (img_name, model, z_dim, patch_size=50, patch_per_img=10, save=True, verbose=False):
    print("Getting features for: " + img_name)
    patch_radius = patch_size // 2   
    img = load_landsat(img_name, bands_only=True)
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
        z = tilenet.encode(patch)
        if cuda: z = z.cpu()
        z = z.data.numpy()
        output += z
    output = output / patch_per_img
    return output

# Setting up model
in_channels = 3
z_dim = 512
cuda = torch.cuda.is_available()
tilenet = make_tilenet(in_channels = 3, z_dim=z_dim)
if cuda: tilenet.cuda()

# Load parameters
model_fn = '/home/asamar/tile2vec/models/TileNet.ckpt'
tilenet.load_state_dict(torch.load(model_fn))
tilenet.eval()
test_imgs = 642
patches_per_img = 10
X = np.zeros((test_imgs, z_dim))

for i in range(test_imgs):
    img_name = '/home/asamar/tile2vec/data/uganda_landsat_test/landsat7_uganda_3yr_cluster_' + str(i) + '.tif'
    X[i] = get_test_features (img_name, tilenet, z_dim, patch_size=50, patch_per_img=10, save=True, verbose=False)

np.save('/home/asamar/tile2vec/data/uganda_lsms/cluster_conv_features.npy', X)
