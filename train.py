# copied from ex2notebook
# with paths changed

import sys
import os
import torch
from torch import optim
from time import time
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import gdal
import random
import scipy.stats as stats
import numpy as np
from fig_utils import *
tile2vec_dir = '/home/asamar/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)

from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet
from src.training import train, validate, prep_triplets
from utils import *

# Environment stuff
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

# Change these arguments to match your directory and desired parameters
img_type = 'landsat'
train_dir = '/home/asamar/tile2vec/data/uganda_patches_train/'
test_dir = '/home/asamar/tile2vec/data/uganda_patches_test/'
bands = 5
augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets_train = 100
n_triplets_test = 100

train_dataloader = triplet_dataloader(img_type, train_dir, bands=bands, augment=augment,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, n_triplets=n_triplets_train, pairs_only=True)

print('Train Dataloader set up complete.')

test_dataloader = triplet_dataloader(img_type, test_dir, bands=bands, augment=augment,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, n_triplets=n_triplets_test, pairs_only=True)

print('Test Dataloader set up complete.')

in_channels = bands
z_dim = 512
TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
TileNet.train()
if cuda: TileNet.cuda()
print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))

epochs = 50
margin = 10
l2 = 0.01
print_every = 10000
save_models = True

model_dir = '/home/asamar/tile2vec/models/'
if not os.path.exists(model_dir): os.makedirs(model_dir)

print('Begin Training')
t0 = time()
train_loss = []
test_loss = []

# regression variables
test_imgs = 642
patches_per_img = 10
country = 'uganda'
country_path = '/home/asamar/tile2vec/data/uganda_lsms/'
dimension = None
k = 5
k_inner = 5
points = 10
alpha_low = 1
alpha_high = 5
margin = 0.25
r2_list = []

for epoch in range(0, epochs):
    avg_loss_train = train(
    TileNet, cuda, train_dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    train_loss.append(avg_loss_train)

    avg_loss_test= validate(
    TileNet, cuda, test_dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    test_loss.append(avg_loss_test)

    if save_models:
        model_fn = os.path.join(model_dir, 'TileNet.ckpt')
        torch.save(TileNet.state_dict(), model_fn)
    plt.figure()
    plt.plot(list(range(0,epoch+1)), train_loss, 'b', label='train')
    plt.plot(list(range(0,epoch+1)), test_loss, 'r', label='test')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('Loss', fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig('loss.png')
    plt.clf()

    X = np.zeros((test_imgs, z_dim))
    for i in range(test_imgs):
        img_name = '/home/asamar/tile2vec/data/uganda_landsat_test/landsat7_uganda_3yr_cluster_' + str(i) + '.tif'
        X[i] = get_test_features (img_name, TileNet, z_dim, cuda, patch_size=50, patch_per_img=10, save=True, verbose=False)

    np.save('/home/asamar/tile2vec/data/uganda_lsms/cluster_conv_features.npy', X)

    
    X, y, y_hat, r2 = predict_consumption(country, country_path,
                                dimension, k, k_inner, points, alpha_low,
                                alpha_high, margin)
    print("r2: " + str(r2))
    r2_list.append(r2)
    plt.figure()
    plt.plot(list(range(0,epoch+1)), r2_list, 'b', label='r^2')
    plt.xlabel('Epoch', fontsize=14)
    plt.ylabel('r^2', fontsize=14)
    plt.legend()
    plt.show()
    plt.savefig('r2.png')
    plt.clf()


    
