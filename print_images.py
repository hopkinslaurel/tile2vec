import numpy as np
import matplotlib.pyplot as plt
import gdal
import sys
import os
import paths

for i in range(0,3):
    print("Patch: " + str(i))
    a = np.load(paths.train_tiles + str(i) + 'anchor.npy')
    n = np.load(paths.train_tiles + str(i) + 'neighbor.npy')
    d = np.load(paths.train_tiles + str(i) + 'distant.npy')
    triplet = np.concatenate((a[:,:,:3], n[:,:,:3], d[:,:,:3]), axis=1)
    triplet = np.flip(triplet, 2)
    plt.imsave(paths.fig_dir + str(i) + 'triplet.jpg', np.clip(triplet,0,50)/float(50))

sys.exit()

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

sys.exit()
