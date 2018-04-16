# copied from ex2notebook
# with paths changed

import sys
import os
import torch
from torch import optim
from time import time

tile2vec_dir = '/home/asamar/tile2vec'
sys.path.append('../')
sys.path.append(tile2vec_dir)

from src.datasets import TileTripletsDataset, GetBands, RandomFlipAndRotate, ClipAndScale, ToFloatTensor, triplet_dataloader
from src.tilenet import make_tilenet

from src.training import prep_triplets, train_triplet_epoch

# Environment stuff
os.environ['CUDA_VISIBLE_DEVICES'] = '0'
cuda = torch.cuda.is_available()

# Change these arguments to match your directory and desired parameters
img_type = 'naip'
tile_dir = '/home/asamar/triplets/'
bands = 4
augment = True
batch_size = 50
shuffle = True
num_workers = 4
n_triplets = 100000

dataloader = triplet_dataloader(img_type, tile_dir, bands=bands, augment=augment,batch_size=batch_size, shuffle=shuffle, num_workers=num_workers, n_triplets=n_triplets, pairs_only=True)

print('Dataloader set up complete.')

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

t0 = time()

print('Begin training.................')
for epoch in range(0, epochs):
    (avg_loss, avg_l_n, avg_l_d, avg_l_nd) = train_triplet_epoch(
    TileNet, cuda, dataloader, optimizer, epoch+1, margin=margin, l2=l2,
    print_every=print_every, t0=t0)
    if save_models:
        model_fn = os.path.join(model_dir, 'TileNet.ckpt')
        torch.save(TileNet.state_dict(), model_fn)
