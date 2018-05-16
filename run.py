# run.py
# =============================================================================
# Original code by Neal Jean/Sherrie Wang (see example 2 notebook in tile2vec
# repo). Minor edits and extensions by Anshul Samar. 

import sys
tile2vec_dir = '/home/asamar/tile2vec'
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
from src.tilenet import make_tilenet
from src.training import train_model, validate_model, prep_triplets
from utils import *
import paths
from tensorboardX import SummaryWriter
import argparse
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('-train', action='store_true')
parser.add_argument('-test',  action='store_true')
parser.add_argument('-test_lsms',  action='store_true')
parser.add_argument('-predict', action='store_true')
parser.add_argument('-debug', action='store_true')
parser.add_argument('--model_fn', dest='model_fn')
args = parser.parse_args()
print(args)

# Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

if args.debug:
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True

# Logging
writer = SummaryWriter(paths.log_dir + 'nips')
    
# Data Parameters
img_type = 'landsat'
bands = 5
augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets_train = 100000
n_triplets_test = 10000

if args.train:
    train_dataloader = triplet_dataloader(img_type, paths.train_tiles,
                                          bands=bands, augment=augment,
                                          batch_size=batch_size,
                                          shuffle=shuffle,
                                          num_workers=num_workers,
                                          n_triplets=n_triplets_train,
                                          pairs_only=True)

    print('Train Dataloader set up complete.')

if args.test:
    test_dataloader = triplet_dataloader(img_type, paths.test_tiles,
                                         bands=bands, augment=augment,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         n_triplets=n_triplets_test,
                                         pairs_only=True)

    print('Test Dataloader set up complete.')

if args.test_lsms:
    lsms_dataloader = triplet_dataloader(img_type, paths.lsms_tiles,
                                         bands=bands, augment=augment,
                                         batch_size=batch_size,
                                         shuffle=shuffle,
                                         num_workers=num_workers,
                                         n_triplets=n_triplets_test,
                                         pairs_only=True)

    print('LSMS Dataloader set up complete.')

# Training Parameters
in_channels = bands
z_dim = 512
TileNet = make_tilenet(in_channels=in_channels, z_dim=z_dim)

if args.model_fn:
    TileNet.load_state_dict(torch.load(args.model_fn))

if cuda: TileNet.cuda()
print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))
epochs = 100
margin = 50
l2 = 0.01
print_every = 10000
save_models = True

if not os.path.exists(paths.model_dir): os.makedirs(paths.model_dir)

print('Begin Training')
t0 = time()

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

# Statistics
train_loss = []
test_loss = []
lsms_loss = []
r2_list = {'avg':{},'emax':{},'sample':{}}
mse_list = {'avg':{},'emax':{},'sample':{}}

# Train
for epoch in range(0, epochs):
    if args.train:
        avg_loss_train = train_model(TileNet, cuda, train_dataloader, optimizer,
                                     epoch+1, margin=margin, l2=l2,
                                     print_every=print_every, t0=t0)
        train_loss.append(avg_loss_train)
        writer.add_scalar('loss/train',avg_loss_train, epoch)

    if args.test:
        avg_loss_test= validate_model(TileNet, cuda, test_dataloader, optimizer,
                                      epoch+1, margin=margin, l2=l2,
                                      print_every=print_every, t0=t0)
        test_loss.append(avg_loss_test)
        writer.add_scalar('loss/test',avg_loss_test, epoch)

    if args.test_lsms:
        avg_loss_lsms= validate_model(TileNet, cuda, lsms_dataloader, optimizer,
                                      epoch+1, margin=margin, l2=l2,
                                      print_every=print_every, t0=t0)
        lsms_loss.append(avg_loss_lsms)
        writer.add_scalar('loss/lsms',avg_loss_lsms, epoch)


    if args.train and args.test and args.test_lsms:    
        writer.add_scalars('loss',{"train":avg_loss_train,
                                   "test":avg_loss_test,
                                   "lsms":avg_loss_lsms}, epoch)

    if args.predict:
        # EMax features
        print("Generating LSMS EMax Features")
        img_names = [paths.lsms_images_big + 'landsat7_uganda_3yr_cluster_' \
                     + str(i) + '.tif' for i in range(test_imgs)]
        X = get_emax_features(img_names, TileNet, z_dim, cuda, bands,
                             patch_size=50, patch_per_img=10, save=True,
                             verbose=False, npy=False)
        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        r2_list['emax'][epoch] = []
        mse_list['emax'][epoch] = []
        for i in range(10):
            _, _, _, r2, mse = predict_consumption(country, country_path,
                                                   dimension, k, k_inner,
                                                   points, alpha_low,
                                                   alpha_high,
                                                   regression_margin)

            r2_list['emax'][epoch].append(r2)
            mse_list['emax'][epoch].append(mse)
    
        print("Emax r2: " + str(r2_list['emax'][epoch]))
        print("Emax mse: " + str(mse_list['emax'][epoch]))

        # Sample Features
        print("Generating LSMS Sample Features")
        img_names = [paths.lsms_images + 'landsat7_uganda_3yr_cluster_' \
                     + str(i) + '.tif' for i in range(test_imgs)]
        X = get_sample_features(img_names, TileNet, z_dim, cuda, bands,
                             patch_size=50, patch_per_img=10, save=True,
                             verbose=False, npy=False)
        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        r2_list['sample'][epoch] = []
        mse_list['sample'][epoch] = []
        for i in range(10):
            _, _, _, r2, mse = predict_consumption(country, country_path,
                                                   dimension, k, k_inner,
                                                   points, alpha_low,
                                                   alpha_high,
                                                   regression_margin)

            r2_list['sample'][epoch].append(r2)
            mse_list['sample'][epoch].append(mse)

        print("Sample r2: " + str(r2_list['sample'][epoch]))
        print("Sample mse: " + str(mse_list['sample'][epoch]))
        
        # Avg Features
        print("Generating LSMS Average Features")
        img_names = [paths.lsms_images_big + 'landsat7_uganda_3yr_cluster_' \
                     + str(i) + '.tif' for i in range(test_imgs)]
        X = get_avg_features(img_names, TileNet, z_dim, cuda, bands,
                             patch_size=50, patch_per_img=10, save=True,
                             verbose=False, npy=False)
        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        r2_list['avg'][epoch] = []
        mse_list['avg'][epoch] = []
        for i in range(10):
            _, _, _, r2, mse = predict_consumption(country, country_path,
                                                   dimension, k, k_inner,
                                                   points, alpha_low,
                                                   alpha_high,
                                                   regression_margin)

            r2_list['avg'][epoch].append(r2)
            mse_list['avg'][epoch].append(mse)

        print("Avg r2: " + str(r2_list['avg'][epoch]))
        print("Avg mse: " + str(mse_list['avg'][epoch]))


        writer.add_scalars('r2',{"sample": np.mean(r2_list['sample'][epoch]),
                                 "emax": np.mean(r2_list['emax'][epoch]),
                                 "avg": np.mean(r2_list['avg'][epoch])}, epoch)

        writer.add_scalars('mse',{"sample": np.mean(mse_list['sample'][epoch]),
                                 "emax": np.mean(mse_list['emax'][epoch]),
                                 "avg": np.mean(mse_list['avg'][epoch])}, epoch)
            
    if save_models:
        save_name = 'TileNet' + str(epoch) + '.ckpt'
        model_path = os.path.join(paths.model_dir, save_name)
        torch.save(TileNet.state_dict(), model_path)
        if args.train:
            with open('train_loss.p', 'wb') as f:
                pickle.dump(train_loss, f)
        if args.test:
            with open('test_loss.p', 'wb') as f:
                pickle.dump(test_loss, f)
        if args.test_lsms:
            with open('lsms_loss.p', 'wb') as f:
                pickle.dump(lsms_loss, f)
        if args.predict:
            with open('r2_' + str(epoch) + '.p', 'wb') as f:
                pickle.dump(r2_list, f)
            with open('mse_' + str(epoch) + '.p', 'wb') as f:
                pickle.dump(mse_list, f)
        


    


    
