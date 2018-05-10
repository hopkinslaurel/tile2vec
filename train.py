# train.py
# =============================================================================
# Original code by Neal Jean/Sherrie Wang (see example 2 notebook in tile2vec
# repo. Minor edits and extensions by Anshul Samar. 

import sys
import os
import torch
from torch import optim
from time import time
import random
import numpy as np
from fig_utils import *
tile2vec_dir = '/home/asamar/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)
from src.datasets import triplet_dataloader
from src.minires import make_minires
from src.training import train, validate, prep_triplets
from utils import *
import paths
from tensorboardX import SummaryWriter

torch.manual_seed(1)

# Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()
if cuda:
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True

# Logging
writer = SummaryWriter(paths.log_dir + 'exp1')
    
# Data Parameters
img_type = 'landsat'
bands = 5
augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets_train = 100
n_triplets_test = 100

train_dataloader = triplet_dataloader(img_type, paths.train_tiles, bands=bands,
                                      augment=augment,batch_size=batch_size,
                                      shuffle=shuffle, num_workers=num_workers,
                                      n_triplets=n_triplets_train, pairs_only=True)

print('Train Dataloader set up complete.')

test_dataloader = triplet_dataloader(img_type, paths.test_tiles, bands=bands,
                                     augment=augment,batch_size=batch_size,
                                     shuffle=shuffle, num_workers=num_workers,
                                     n_triplets=n_triplets_test, pairs_only=True)

print('Test Dataloader set up complete.')

lsms_dataloader = triplet_dataloader(img_type, paths.lsms_tiles, bands=bands,
                                     augment=augment,batch_size=batch_size,
                                     shuffle=shuffle, num_workers=num_workers,
                                     n_triplets=n_triplets_test, pairs_only=True)

print('LSMS Dataloader set up complete.')

# Training Parameters
in_channels = bands
z_dim = 256
# TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)
# TileNet = make_cnn(in_channels=in_channels, z_dim=z_dim)
TileNet = make_minires(in_channels=in_channels)
# TileNet.train()
# model_fn = '/home/asamar/tile2vec/models/5_bands/TileNet.ckpt'
# TileNet.load_state_dict(torch.load(model_fn))
if cuda: TileNet.cuda()
print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 50
margin = 10
l2 = 0.01
print_every = 10000
save_models = True

if not os.path.exists(paths.model_dir): os.makedirs(paths.model_dir)

print('Begin Training')
t0 = time()
# train_loss = []
# test_loss = []
# lsms_loss = []

# Regression variables
lsms = True
test_imgs = 642
patches_per_img = 10
country = 'uganda'
country_path = paths.lsms_data
dimension = None
k = 5
k_inner = 5
points = 10
alpha_low = 1
alpha_high = 5
regression_margin = 0.25
# r2_list = []
# mse_list = []

# Train
for epoch in range(0, epochs):
    avg_loss_train = train(
    TileNet, cuda, train_dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    # train_loss.append(avg_loss_train)

    avg_loss_test= validate(
    TileNet, cuda, test_dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    # test_loss.append(avg_loss_test)

    avg_loss_lsms= validate(
        TileNet, cuda, lsms_dataloader, optimizer, epoch+1, margin=margin, l2=l2,
        print_every=print_every, t0=t0)
    # lsms_loss.append(avg_loss_lsms)

    writer.add_scalars('loss',{"train":avg_loss_train,
                               "test":avg_loss_test,
                               "lsms":avg_loss_lsms}, epoch)
    
    if save_models:
        model_fn = os.path.join(paths.model_dir, 'TileNet.ckpt')
        torch.save(TileNet.state_dict(), model_fn)

    if lsms:
        TileNet.eval()
        X = np.zeros((test_imgs, z_dim))
        for i in range(test_imgs):
            img_name = paths.lsms_images + 'landsat7_uganda_3yr_cluster_' + str(i) + '.tif'
            X[i] = get_test_features (img_name, TileNet, z_dim, cuda, bands,
                                  patch_size=50, patch_per_img=10, save=True,
                                  verbose=False, npy=False)

        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        X, y, y_hat, r2, mse = predict_consumption(country, country_path,
                                                   dimension, k, k_inner, points, alpha_low,
                                                   alpha_high, regression_margin)
        print("r2: " + str(r2))
        print("mse: " + str(mse))
        # r2_list.append(r2)
        # mse_list.append(mse)
        writer.add_scalar('r2', r2, epoch)
        writer.add_scalar('mse', mse, epoch)


    


    
