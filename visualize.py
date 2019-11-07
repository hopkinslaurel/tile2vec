import numpy as np
import matplotlib.pyplot as plt
import sys
import os
import paths
import gdal
import torch
import torchvision.utils as vutils
from fig_utils import load_country_lsms
import math
from tensorboardX import SummaryWriter

writer_data = SummaryWriter(paths.log_dir + 'data')
writer_lsms = SummaryWriter(paths.log_dir + 'lsms')

def load_image(filename):
    img_bgr = np.load(filename)[:,:,:3]  # 50x50x3 images
    img_rgb = np.flip(img_bgr, 2)
    img_channel_front = np.moveaxis(img_rgb, -1, 0)
    print('front shape: ')
    print(img_channel_front.shape)
    print('front max: ')
    #print(img_channel_front.amax)
    one = img_channel_front[2]
    #print(np.amax(img_channel_front,axis=0))
    print(np.amax(one, axis=0))
    print('front min: ')
    print(np.amin(one, axis=0))
    img_clip = np.clip(img_channel_front,0,50)/float(50)
    
    return np.expand_dims(img_clip,0)

def visualize_raw(num_images, img_path, tag):
    for i in range(0, num_images):
        a = load_image(img_path + str(i) + 'anchor.npy')
        n = load_image(img_path + str(i) + 'neighbor.npy')
        d = load_image(img_path + str(i) + 'distant.npy')
        triplet = np.concatenate((a, n, d), axis=0)
        x = vutils.make_grid(torch.from_numpy(triplet), 3)
        writer_data.add_image(tag + "/" + str(i),x)

visualize_raw(10, paths.train_tiles, 'train')
visualize_raw(10, paths.test_tiles, 'test')
#visualize_raw(10, paths.lsms_train_tiles, 'lsms_train')
#visualize_raw(10, paths.lsms_test_tiles, 'lsms_test')

def visualize_lsms(num_images, img_path, tag, X, y, max_len):
    data = torch.zeros(num_images, 3, max_len, max_len)
    for i in range(0, num_images):
        obj = gdal.Open(img_path + 'landsat7_uganda_3yr_cluster_' + str(i) + '.tif')
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset
        img = np.flip(img[:3,:max_len,:max_len],0)
        img = np.clip(img,0,50)/float(50)
        #plt.imsave(str(i) + '.jpg', np.moveaxis(img, 0,2))
        data[i] = torch.from_numpy(img)
        writer_data.add_image("lsms/" + str(i),torch.from_numpy(img))
        writer_lsms.add_embedding(X[:num_images], metadata=y[:num_images],
                              label_img=data, tag=tag)

#X, _, y = load_country_lsms(paths.lsms_data)
#X = torch.from_numpy(X)
#y = np.ndarray.tolist(y)
#visualize_lsms(10, paths.lsms_images, 'last_exp', X, y, 74)
                       
def sanity():
    for i in range(0,3):
        print("Patch: " + str(i))
        a = np.load(paths.train_tiles + str(i) + 'anchor.npy')
        n = np.load(paths.train_tiles + str(i) + 'neighbor.npy')
        d = np.load(paths.train_tiles + str(i) + 'distant.npy')
        triplet = np.concatenate((a[:,:,:3], n[:,:,:3], d[:,:,:3]), axis=1)
        triplet = np.flip(triplet, 2)
        print('triplet')
        print(triplet)
        plt.imsave(paths.fig_dir + str(i) + 'triplet.jpg',
                np.clip(triplet,0,50)) #/float(50))
'''
    for filename in os.listdir(test_dir):
        if filename.endswith('.tif') and "cluster" in filename:
            print(filename)
            obj = gdal.Open(tif_dir + filename)
            img = obj.ReadAsArray().astype(np.uint8)
            del obj # close GDAL dataset
            img = np.moveaxis(img, 0, -1)
            img = img[:,:,:3]
            img = np.flip(img, 2)
            plt.imsave(img_dir + filename + '.jpg', np.clip(img,0,50)/float(50))
'''
sanity()
