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
parser.add_argument('-predict_small', action='store_true')
parser.add_argument('-predict_big', action='store_true')
parser.add_argument('-quantile', action='store_true')
parser.add_argument('-debug', action='store_true')
parser.add_argument('--model_fn', dest='model_fn')
parser.add_argument('--exp_name', dest='exp_name')
parser.add_argument('--epochs', dest="epochs", type=int, default=50)
parser.add_argument('--z_dim', dest="z_dim", type=int, default=512)
parser.add_argument('--trials', dest="trials", type=int, default=10)
parser.add_argument('-save_models', action='store_true')
parser.add_argument('--model', dest="model", default="tilenet")


args = parser.parse_args()
print(args)
if args.model == "minires":
    from src.minires import make_tilenet
else:
    from src.tilenet import make_tilenet

if not args.save_models:
    print("NOT SAVING CHECKPOINTS")

# Environment
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

if args.debug:
    torch.manual_seed(1)
    if cuda:
        torch.cuda.manual_seed_all(1)
        torch.backends.cudnn.deterministic = True

# Logging
writer = SummaryWriter(paths.log_dir + args.exp_name)
    
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
TileNet = make_tilenet(in_channels=in_channels, z_dim=args.z_dim)
if cuda: TileNet.cuda()

if args.model_fn:
    TileNet.load_state_dict(torch.load(args.model_fn))

print('TileNet set up complete.')

lr = 1e-3
optimizer = optim.Adam(TileNet.parameters(), lr=lr, betas=(0.5, 0.999))
margin = 50
l2 = 0.01
print_every = 10000

if not os.path.exists(paths.model_dir): os.makedirs(paths.model_dir)
if not os.path.exists(paths.model_dir + args.exp_name):
    os.makedirs(paths.model_dir + args.exp_name)

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
r2_list = {'big':{},'small':{}}
mse_list = {'big':{},'small':{}}
save_dir = paths.model_dir + args.exp_name + '/'

# Train
for epoch in range(0, args.epochs):
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

    if args.predict_small:
        # Small Image Features
        print("Generating LSMS Small Features")
        img_names = [paths.lsms_images + 'landsat7_uganda_3yr_cluster_' \
                     + str(i) + '.tif' for i in range(test_imgs)]
        X = get_small_features(img_names, TileNet, args.z_dim, cuda, bands,
                               patch_size=50, patch_per_img=10, save=True,
                               verbose=False, npy=False, quantile=args.quantile)
        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        r2_list['small'][epoch] = []
        mse_list['small'][epoch] = []
        for i in range(args.trials):
            X, y, y_hat, r2, mse = predict_consumption(country, country_path,
                                                       dimension, k, k_inner,
                                                       points, alpha_low,
                                                       alpha_high,
                                                       regression_margin)

            r2_list['small'][epoch].append(r2)
            mse_list['small'][epoch].append(mse)
        with open(save_dir + '/y_small_e' + str(epoch) + '.p', 'wb') as f:
            pickle.dump((y, y_hat, r2),f)
        print("Small r2: " + str(r2_list['small'][epoch]))
        print("Small mse: " + str(mse_list['small'][epoch]))

    if args.predict_big:
        # Big Image Features
        print("Generating LSMS Big Image Features")
        img_names = [paths.lsms_images_big + 'landsat7_uganda_3yr_cluster_' \
                     + str(i) + '.tif' for i in range(test_imgs)]
        X = get_big_features(img_names, TileNet, args.z_dim, cuda, bands,
                             patch_size=50, patch_per_img=10, save=True,
                             verbose=False, npy=False, quantile=args.quantile)
        np.save(paths.lsms_data + 'cluster_conv_features.npy', X)

        r2_list['big'][epoch] = []
        mse_list['big'][epoch] = []
        for i in range(args.trials):
            X, y, y_hat, r2, mse = predict_consumption(country, country_path,
                                                   dimension, k, k_inner,
                                                   points, alpha_low,
                                                   alpha_high,
                                                   regression_margin)

            r2_list['big'][epoch].append(r2)
            mse_list['big'][epoch].append(mse)
        with open(save_dir + '/y_big_e' + str(epoch) + '.p', 'wb') as f:
            pickle.dump((y, y_hat, r2),f)
        print("Big r2: " + str(r2_list['big'][epoch]))
        print("Big mse: " + str(mse_list['big'][epoch]))

    if args.predict_small and args.predict_big:
        writer.add_scalars('r2',{"small": np.mean(r2_list['small'][epoch]),
                                 "big": np.mean(r2_list['big'][epoch])}, epoch)

        writer.add_scalars('mse',{"small": np.mean(mse_list['small'][epoch]),
                                 "big": np.mean(mse_list['big'][epoch])}, epoch)
            
    if args.save_models:
        print("Saving")
        save_name = 'TileNet' + str(epoch) + '.ckpt'
        model_path = os.path.join(save_dir, save_name)
        torch.save(TileNet.state_dict(), model_path)
        if args.train:
            with open(save_dir + '/train_loss.p', 'wb') as f:
                pickle.dump(train_loss, f)
        if args.test:
            with open(save_dir + '/test_loss.p', 'wb') as f:
                pickle.dump(test_loss, f)
        if args.test_lsms:
            with open(save_dir + '/lsms_loss.p', 'wb') as f:
                pickle.dump(lsms_loss, f)
        if args.predict_big or args.predict_small:
            with open(save_dir + '/r2_' + str(epoch) + '.p', 'wb') as f:
                pickle.dump(r2_list, f)
            with open(save_dir + '/mse_' + str(epoch) + '.p', 'wb') as f:
                pickle.dump(mse_list, f)
        
print("Finished.")

    


    
