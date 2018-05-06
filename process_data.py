# process_data.py
# =============================================================================
# Convert a directory of .tif files (i.e. from earth engine) into numpy arrays
# with given width/height. For example, if exported .tif files are 1000x1000
# you can convert them into ten 100x100 npy arrays.
# Example: use process_data.py on all uganda tif files. Then, to do a split:
# use mv `ls | shuf | head -NUM` paths.train_data. Move remaining images to
# paths.test_data. 

import gdal
import numpy as np
import os
import paths

width = 145
height = 145
image_num = 0
for filename in sorted(os.listdir(paths.tif_dir)):
    print(filename)
    if filename.endswith('.tif'):
        obj = gdal.Open(paths.tif_dir + filename)
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset
        img = np.moveaxis(img, 0, -1)
        for i in range(0, img.shape[0] // width):
            for j in range(0, img.shape[1] // height):
                start_r = i*width
                end_r = (i+1)*width
                start_c = j*height
                end_c = (j+1)*height
                save_as = paths.np_dir + "/" + str(image_num)+".npy"
                np.save(save_as, img[start_r:end_r,start_c:end_c,:])
                image_num += 1

