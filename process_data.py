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
import argparse
import matplotlib.pyplot as plt  
import utils

# Parse command line arguments
parser = argparse.ArgumentParser()

# Directory to chunk (naip or nlcd?) -- only need to chunk naip for original T2V, need both for T2V+ ('smarter' similarity metric)
parser.add_argument('--data', dest='data', type=str, default='naip')
parser.add_argument('-clip_center', action='store_true')

args = parser.parse_args()
print(args)

if args.data == 'naip':
    tif_path = paths.tif_path
    npy_path = paths.np_path
elif args.data == 'nlcd':
    tif_path = paths.tif_dir_nlcd
    npy_path = paths.np_dir_nlcd
print("Chunking " + args.data + " images") 

width = 200 #145  # patch size, from which triplet tiles will be taken
height = 200 #145 #500*(10m/pixel) 5.5km patches, for 200 x 200 pixel (200*10m/pixel) 2km tiles 
image_num = 0
for filename in sorted(os.listdir(tif_path)):
    print(filename)
    if filename.endswith('.tif'):
        obj = gdal.Open(tif_path + filename)
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset
    if filename.endswith('.npy'):
        img = np.load(tif_path + filename)
    if args.data == 'naip':
        img = np.moveaxis(img, 0, -1)  # (3, x_dim, y_dim) -> (x_dim,y_dim, 3)
        #print(img.shape[0], img.shape[1])
    if args.clip_center:
        print(img.shape)
        center = (int(img.shape[0]/2), int(img.shape[1]/2))
        start_r = center[0]-int(height/2)
        end_r = center[0]+int(height/2)
        start_c = center[1]-int(width/2)
        end_c = center[1]+int(width/2)

        save_as = npy_path + filename.split('.tif')[0] +".npy"
        if args.data == 'nlcd':
            #plt.imsave(args.data + '_tile' + str(image_num) + '.tiff', img[start_r:end_r,start_c:end_c])
            np.save(save_as, img[start_r:end_r,start_c:end_c])
        else:
            #plt.imsave(args.data + '_tile' + str(image_num) + '.jpg', img[start_r:end_r,start_c:end_c,:])
            np.save(save_as, img[start_r:end_r,start_c:end_c,:])
    else:
        for i in range(0, img.shape[0] // width):
            for j in range(0, img.shape[1] // height): 
                start_r = i*width
                end_r = (i+1)*width
                start_c = j*height
                end_c = (j+1)*height

                save_as = npy_path + filename.split('.')[0] +".npy"
                if args.data == 'nlcd':
                    #plt.imsave(args.data + '_tile' + str(image_num) + '.tiff', img[start_r:end_r,start_c:end_c])
                    np.save(save_as, img[start_r:end_r,start_c:end_c])
                else:
                    #plt.imsave(args.data + '_tile' + str(image_num) + '.jpg', img[start_r:end_r,start_c:end_c,:])
                    np.save(save_as, img[start_r:end_r,start_c:end_c,:])           
                
                # test image that was saved 
                #img_array = np.load(save_as)
                #plt.imsave(paths.fig_dir_test_load + str(image_num) + '_load.jpg', img_array)
                
                image_num += 1
