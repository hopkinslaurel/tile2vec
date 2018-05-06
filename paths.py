home_dir = "/home/asamar/tile2vec/"

# for process_data.py
data_dir = home_dir + "/data/"
tif_dir = data_dir + "tif/uganda_all/" # earth engine exports
np_dir = data_dir + "npy/uganda/" # output dir for process_data

# for make_triplets.py
train_images = data_dir + "npy/train/" # contains train images
test_images = data_dir + "npy/test/" # contains test images
lsms_images = data_dir + "tif/uganda_lsms/" # contains lsms images
train_tiles = data_dir + "tiles/train/" # train tiles (triplets)
test_tiles = data_dir + "tiles/test/" # test tiles (triplets)
lsms_tiles = data_dir + "tiles/lsms/" # lsms tiles (triplets)

# for figures
fig_dir = home_dir + "/figures/"

# model dir
model_dir = home_dir + "/models/"

# for regression
lsms_data = home_dir + "/lsms/uganda_lsms/"
