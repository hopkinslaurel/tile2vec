# Based on Neal Jean/Sherrie's code

import sys
sys.path.append('/home/asamar/tile2vec/')
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
import argparse
import paths

parser = argparse.ArgumentParser()
parser.add_argument('--model_fn', dest='model_fn')
parser.add_argument('-big', action='store_true')
parser.add_argument('-small', action='store_true')
parser.add_argument('-q', action='store_true')
args = parser.parse_args()
print(args)

# Setting up model
in_channels = 5
z_dim = 512
cuda = torch.cuda.is_available()
tilenet = make_tilenet(in_channels = in_channels, z_dim=z_dim)
if cuda: tilenet.cuda()

# Load parameters
tilenet.load_state_dict(torch.load(args.model_fn))
tilenet.eval()

test_imgs = 642
patches_per_img = 10
bands = 5

# See code in run.py for remaining code to generate features


