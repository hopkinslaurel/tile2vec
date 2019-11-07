home_dir = "/home/hopkilau/_tile2vec/"
drive = "/datadrive/tile2vec/"

# for process_data.py
data_dir = home_dir + "data/"
tif_dir = data_dir + "tif/unprocessed_naip/"  # earth engine exports
np_dir = data_dir + "npy/oregon/" # output dir for process_data

# for make_triplets.py
all_images = data_dir + "npy/all/" # contains all images
train_images = data_dir + "npy/train/" # contains train images
test_images = data_dir + "npy/test/" # contains test images
lsms_images_small = data_dir + "tif/ebird_small/" # contains lsms images
lsms_images_big = data_dir + "tif/uganda_lsms_big/" # contains big images
train_tiles = drive + "data/tiles/train/"  #data_dir + "tiles/train/" # train tiles (triplets)
val_tiles = drive + "data/tiles/val/"  #data_dir + "tiles/val/" # val tiles (triplets)
test_tiles = drive + "data/tiles/test/"  #data_dir + "tiles/test/" # test tiles (triplets)
lsms_train_tiles = drive + "data/tiles/ebird_train/"  #data_dir + "tiles/eBird_train/" # lsms tiles (triplets)
lsms_val_tiles = drive + "data/tiles/ebird_val/"  #data_dir + "tiles/ebird_val/" # lsms tiles (triplets)

# figure dir
fig_dir = home_dir + "figures/"
fig_dir_test = home_dir + "test_figs/"
fig_dir_test_load = home_dir + "test_figs_load/"
fig_dir_test_land = home_dir + "test_figs_land/"

# log dir
log_dir = drive + "/runs/"

# model dir
model_dir = home_dir + "/models/"

# for regression
lsms_data = home_dir + "/lsms/uganda_lsms/"
original_data = home_dir + "/lsms/original_lsms/"

# for extracting features
ebird_features = drive + "data/ebird_features/"
