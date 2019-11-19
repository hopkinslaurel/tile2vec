# process_data.py
# =============================================================================
# Convert a directory of .tif files (i.e. from earth engine) into numpy arrays
# with given width/height. For example, if exported .tif files are 1000x1000
# you can convert them into ten 100x100 npy arrays.
# Example: use process_data.py on all uganda tif files. Then, to do a split:
# use mv `ls | shuf | head -NUM` [dest_folder]. Move remaining images to
# the other folder. 

import gdal
import numpy as np
import os
import paths
import matplotlib.pyplot as plt  # LH

width = 145  # patch size, from which triplet tiles will be taken
height = 145 #145*(30m/pixel) should be ~4.4km x 4.4km images 
image_num = 0
for filename in sorted(os.listdir(paths.tif_dir)):
    if filename.endswith('.tif'):
        print(filename)
        obj = gdal.Open(paths.tif_dir + filename)
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset
        img = np.moveaxis(img, 0, -1)
        print(img.shape[0], img.shape[1])
        #for i in range(0, 3):  # LH
        for i in range(0, img.shape[0] // width):
            #for j in range(0, 3): # LH
            for j in range(0, img.shape[1] // height): 
                start_r = i*width
                end_r = (i+1)*width
                start_c = j*height
                end_c = (j+1)*height
                #plt.imsave('naip_tile' + str(image_num) + '.jpg', img[start_r:end_r,start_c:end_c,:])
                                
                save_as = paths.np_dir + "/" + str(image_num)+".npy"
                np.save(save_as, img[start_r:end_r,start_c:end_c,:])
                
                # test image that was saved 
                #img_array = np.load(save_as)
                #plt.imsave(paths.fig_dir_test_load + str(image_num) + '_load.jpg', img_array)
                
                image_num += 1

