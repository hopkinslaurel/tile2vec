# copied from ex3 notebook
# with paths changed

import numpy as np
import os
import torch
from time import time
from torch.autograd import Variable
from src.tilenet import make_tilenet
from src.resnet import ResNet18
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Setting up model
in_channels = 4
z_dim = 512
cuda = torch.cuda.is_available()
tilenet = ResNet18()
if cuda: tilenet.cuda()

# Load parameters
model_fn = '/home/asamar/tile2vec/models/naip_trained.ckpt'
tilenet.load_state_dict(torch.load(model_fn))
tilenet.eval()

# Get data
tile_dir = '/home/asamar/tile2vec/data/tiles'
n_tiles = 1000
y = np.load(os.path.join(tile_dir, 'y.npy'))
print(y.shape)

# Embed tiles
t0 = time()
X = np.zeros((n_tiles, z_dim))
for idx in range(n_tiles):
    tile = np.load(os.path.join(tile_dir, '{}tile.npy'.format(idx+1)))
    # Get first 4 NAIP channels (5th is CDL mask)
    tile = tile[:,:,:4]
    # Rearrange to PyTorch order
    tile = np.moveaxis(tile, -1, 0)
    tile = np.expand_dims(tile, axis=0)
    # Scale to [0, 1]
    tile = tile / 255
    # Embed tile
    tile = torch.from_numpy(tile).float()
    tile = Variable(tile)
    if cuda: tile = tile.cuda()
    z = tilenet.encode(tile)
    if cuda: z = z.cpu()
    z = z.data.numpy()
    X[idx,:] = z
t1 = time()
print('Embedded {} tiles: {:0.3f}s'.format(n_tiles, t1-t0))

# Check CDL classes
print(set(y))

# Reindex CDL classes
y = LabelEncoder().fit_transform(y)
print(set(y))

n_trials = 1000
accs = np.zeros((n_trials,))
for i in range(n_trials):
    # Splitting data and training RF classifer
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2)
    rf = RandomForestClassifier()
    rf.fit(X_tr, y_tr)
    accs[i] = rf.score(X_te, y_te)
print('Mean accuracy: {:0.4f}'.format(accs.mean()))
print('Standard deviation: {:0.4f}'.format(accs.std()))
