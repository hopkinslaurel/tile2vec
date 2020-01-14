home_dir = "/home/hopkilau/_tile2vec/"
drive = "/datadrive/tile2vec/"
mnt = "/mnt/tile2vec/"

# for process_data.py
data_dir = home_dir + "data/"
#tif_dir =  data_dir + "tif/unprocessed_naip/"  # earth engine exports
tif_dir = drive + "data/unprocessed_naip_10m/"
#np_dir = data_dir + "npy/oregon/" # output dir for process_data
np_dir = drive + "data/npy/oregon_10m"

# for make_triplets.py
#all_images = data_dir + "npy/all/" # contains all images
all_images = drive + "data/npy_10m/all/"
#train_images = data_dir + "npy/train/" # contains train images
train_images = drive + "data/npy_10m/train/"  
#test_images = data_dir + "npy/test/" # contains test images
test_images = drive + "data/npy_10m/test/"
#ebird_tifs = data_dir + "tif/ebird_naip/" # contains images centered at ebird records
ebird_tifs = data_dir + "tif/synthetic_5k_naip_mini/"  # contains images centered at randomly sampled points
lsms_images_big = data_dir + "tif/uganda_lsms_big/" # contains big images
#train_tiles = data_dir + "tiles/train/"  #"data/tiles/train/"  # train tiles (triplets)
train_tiles = drive + "data/tiles_10m/train/"
#val_tiles = data_dir + "tiles/val/"  # drive + "data/tiles/val/"  # val tiles (triplets)
val_tiles = drive + "data/tiles_10m/val/"
#test_tiles = data_dir + "tiles/test/"  # drive + "data/tiles/test/"  # test tiles (triplets)
test_tiles = drive + "data/tiles_10m/test/"
lsms_train_tiles = drive + "data/tiles/ebird_train/"  #data_dir + "tiles/eBird_train/" # lsms tiles (triplets)
lsms_val_tiles = drive + "data/tiles/ebird_val/"  #data_dir + "tiles/ebird_val/" # lsms tiles (triplets)

# figure dir
fig_dir = home_dir + "figures/"
fig_dir_test = home_dir + "test_figs/"
fig_dir_test_load = home_dir + "test_figs_load/"
fig_dir_test_land = home_dir + "test_figs_land/"

# log dir
log_dir = drive + "runs/"

# model dir
model_dir = drive + "models/"

# for regression
lsms_data = home_dir + "/lsms/uganda_lsms/"
original_data = home_dir + "/lsms/original_lsms/"

# for extracting features
#ebird_features = drive + "data/ebird_features/"
ebird_features = drive + "data/synthetic_5k_naip_clipped/"

