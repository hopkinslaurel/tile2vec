# sample_triplets.py
# =============================================================================
# Original code by Neal Jean: nealjean/pixel2vec/notebooks/NJ5_naip_sampling*
# Samples triplets (anchor, neighbor, distant) from folder of tif or npy
# images. Anchor and neighbor from same image file, distant from different
# file. Minor extensions/edits by Anshul Samar. 

import numpy as np
import os
import random
from time import time
from utils import *
import paths

def get_triplet_imgs(imgs_dir, n_triplets=1000, no_duplicates=False):
    """
    Create n_triplet random pairs of image filenames.
    Anchor/neighbor patches will come from first file, 
    distant patch from second file.
    imgs_dir: folder of .tif landsat image files
    """
    img_names = []
    for filename in os.listdir(imgs_dir):
        if filename.endswith('.tif'):
            img_names.append(filename)
    # hack - adding 1000 to account for possible duplicates
    img_triplets = list(map(lambda _: random.choice(img_names),
                            range(2 * n_triplets + 1000)))
    img_triplets = np.array(img_triplets)
    img_triplets = img_triplets.reshape((-1, 2))
    if no_duplicates:
        delete = []
        for idx, row in enumerate(img_triplets):
            if row[0] == row[1]:
                delete.append(idx)
        img_triplets = np.delete(img_triplets, delete, 0)
        if len(delete) > 1000:
            print("Not enough triplets!")

    # filtering out extra
    return img_triplets[:n_triplets]
        

def get_patch_triplets(patch_dir, img_dir, img_triplets, bands=7, patch_size=50,
                       neighborhood=125, save=True, verbose=False):
    """
    For each unique image, extracts any anchor
    /neighbor/distant patches as specified by
    img_triplets. 
    Returns "patches" - n_tripletsx3x2 array of x,y
    coordinates of the triplets.
    Saves respective images to patch_dir.
    Assumes input in form (w,h,c)
    """
    t0 = time()
    if not os.path.exists(patch_dir):
        os.makedirs(patch_dir)
    size_even = (patch_size % 2 == 0)
    patch_radius = patch_size // 2

    n_triplets = img_triplets.shape[0]
    unique_imgs = np.unique(img_triplets)
    total_imgs = len(unique_imgs)
    patches = np.zeros((n_triplets, 3, 2), dtype=np.int16)

    save_count = 0
    for img_count, img_name in enumerate(unique_imgs):
        #print('\nStarting image {}/{}: {:0.3f}s\n'.format(img_count+1, total_imgs, time()-t0))
        print("Sampling image {}".format(img_name))
        img = load_landsat(img_dir+img_name, bands, bands_only=True)
        img_padded = np.pad(img, pad_width=[(patch_radius, patch_radius),
                                            (patch_radius, patch_radius), (0,0)],
                            mode='reflect')
        img_shape = img_padded.shape

        for idx, row in enumerate(img_triplets):
            if row[0] == img_name:
                xa, ya = sample_anchor(img_shape, patch_radius)
                xn, yn = sample_neighbor(img_shape, xa, ya, neighborhood, patch_radius)
                
                if verbose:
                    print("    Saving anchor and neighbor patch #{}".format(idx))
                    print("    Anchor patch center:{}".format((xa, ya)))
                    print("    Neighbor patch center:{}".format((xn, yn)))
                if save:
                    patch_anchor = extract_patch(img_padded, xa, ya, patch_radius)
                    patch_neighbor = extract_patch(img_padded, xn, yn, patch_radius)
                    if size_even:
                        patch_anchor = patch_anchor[:-1,:-1]
                        patch_neighbor = patch_neighbor[:-1,:-1]
                    np.save(os.path.join(patch_dir, '{}anchor.npy'.format(idx)), patch_anchor)
                    np.save(os.path.join(patch_dir, '{}neighbor.npy'.format(idx)), patch_neighbor)
                    save_count += 2
                
                patches[idx,0,:] = xa, ya
                patches[idx,1,:] = xn, yn
                
                if row[1] == img_name:
                    # distant image is same as anchor/neighbor image
                    xd, yd = sample_distant_same(img_shape, xa, ya, neighborhood, patch_radius)
                    if verbose:
                        print("    Saving distant patch #{}".format(idx))
                        print("    Distant patch center:{}".format((xd, yd)))
                    if save:
                        patch_distant = extract_patch(img_padded, xd, yd, patch_radius)
                        if size_even:
                            patch_distant = patch_distant[:-1,:-1]
                        np.save(os.path.join(patch_dir, '{}distant.npy'.format(idx)), patch_distant)
                        save_count += 1
                    patches[idx,2,:] = xd, yd
            
            elif row[1] == img_name: 
                # distant image is different from anchor/neighbor image
                xd, yd = sample_distant_diff(img_shape, patch_radius)
                if verbose:
                        print("    Saving distant patch #{}".format(idx))
                        print("    Distant patch center:{}".format((xd, yd)))
                if save:
                    patch_distant = extract_patch(img_padded, xd, yd, patch_radius)
                    if size_even:
                        patch_distant = patch_distant[:-1,:-1]
                    np.save(os.path.join(patch_dir, '{}distant.npy'.format(idx)), patch_distant)
                    save_count += 1
                patches[idx,2,:] = xd, yd
        #print('\nFinished image {}/{}: {} patches saved\n'.format(img_count+1, total_imgs, save_count))
            
    return patches

def sample_anchor(img_shape, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xa = np.random.randint(0, w) + patch_radius
    ya = np.random.randint(0, h) + patch_radius
    return xa, ya

def sample_neighbor(img_shape, xa, ya, neighborhood, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xn = np.random.randint(max(xa-neighborhood, patch_radius),
                           min(xa+neighborhood, w+patch_radius))
    yn = np.random.randint(max(ya-neighborhood, patch_radius),
                           min(ya+neighborhood, h+patch_radius))
    return xn, yn

def sample_distant_same(img_shape, xa, ya, neighborhood, patch_radius):
    w_padded, h_padded, c = img_shape
    w = w_padded - 2 * patch_radius
    h = h_padded - 2 * patch_radius
    
    xd, yd = xa, ya
    while (xd >= xa - neighborhood) and (xd <= xa + neighborhood):
        xd = np.random.randint(0, w) + patch_radius
    while (yd >= ya - neighborhood) and (yd <= ya + neighborhood):
        yd = np.random.randint(0, h) + patch_radius
    return xd, yd

def sample_distant_diff(img_shape, patch_radius):
    return sample_anchor(img_shape, patch_radius)


# Run
np.random.seed(1)
bands = 7
data_dir = '/home/asamar/tile2vec/data/uganda_landsat/'
print("Generating Train Set")
#img_triplets = get_triplet_imgs(data_dir,100000)
#print(img_triplets)
#patches_train = get_patch_triplets('/home/asamar/tile2vec/data/uganda_patches_train/',
#                                   data_dir, img_triplets, bands, patch_size =50,
#                                   neighborhood=125, save=True,verbose=False)

print("Generating Test Set")
#img_triplets = get_triplet_imgs(data_dir,10000)
#print(img_triplets)
#patches_test = get_patch_triplets('/home/asamar/tile2vec/data/uganda_patches_test/',
#                                  data_dir, img_triplets, bands, patch_size =50,
#                                  neighborhood=125, save=True,verbose=False)

print("Generating LSMS Set")
data_dir = '/home/asamar/tile2vec/data/uganda_landsat_test/'
img_triplets = get_triplet_imgs(data_dir,10000,no_duplicates=True)
print(img_triplets)
patches_test = get_patch_triplets('/home/asamar/tile2vec/data/uganda_patches_lsms/',
                                  data_dir, img_triplets, bands, patch_size =50,
                                  neighborhood=125, save=True,verbose=False)



                    
