import numpy as np
import matplotlib.pyplot as plt
import gdal
import os

tif_dir = '/home/asamar/tile2vec/data/uganda_landsat/'
data_dir = '/home/asamar/tile2vec/data/uganda_patches/'
img_dir = '/home/asamar/tile2vec/data/uganda_samples/'

for filename in os.listdir(tif_dir):
    continue
    if filename.endswith('.tif') and "row_0" in filename:
        print(filename)
        obj = gdal.Open(tif_dir + filename)
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset
        img = np.moveaxis(img, 0, -1)
        img = img[:,:,:3]
        img = np.flip(img, 2)
        plt.imsave(img_dir + filename + '.jpg', np.clip(img,0,50)/float(50))

for i in range(0,3):
    print("Patch: " + str(i))
    a = np.load(data_dir + str(i) + 'patch.npy')
    n = np.load(data_dir + str(i) + 'neighbor.npy')
    d = np.load(data_dir + str(i) + 'distant.npy')
    triplet = np.concatenate((a, n, d), axis=1)
    triplet = np.flip(triplet, 2)
    plt.imsave(img_dir + str(i) + 'triplet.jpg', np.clip(triplet,0,50)/float(50))
    
