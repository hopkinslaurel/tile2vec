import cv2
import numpy as np
import gdal
import matplotlib.pyplot as plt
import os
import paths


img_path = paths.to_resize
for filename in sorted(os.listdir(img_path)):
    if filename.endswith('.npy'):
        print(filename)
        #img = cv2.imread(img_path + filename)
        img = np.load(img_path + filename)
        #print(img)
        #print(img.shape)
        res = cv2.resize(img, dsize=(100, 100), interpolation=cv2.INTER_LINEAR)
        #print(res.shape)
        save_as =  paths.resized + filename
        np.save(save_as, res)

    if filename.endswith('.tif') or filename.endswith('.tiff'):
        print(filename)
        obj = gdal.Open(img_path + filename)
        img = obj.ReadAsArray().astype(np.uint8)
        del obj # close GDAL dataset    
        if "naip" in filename:
            #print(img.shape)
            if img.shape[0] == 4:
                img = np.delete(img,3,0) # if image has been clipped, an odd 4th dimension of all 255 is appeneded
            img = np.moveaxis(img, 0, -1)
        #print(img.shape)
        res = cv2.resize(img, dsize=(100,100), interpolation=cv2.INTER_LINEAR)
        #print(res.shape)
        #print(res)
        filename_no_ext = os.path.splitext(filename)[0]
        new_filename = filename_no_ext.replace("10m", "20m") # replace 10m w/20m and remove .tif
        save_as = paths.resized + new_filename
        np.save(save_as, res)
                
        #plt.imsave(args.data + '_tile' + str(image_num) + '.tiff', img[start_r:end_r,start_c:end_c])
