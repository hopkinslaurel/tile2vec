home_dir = "/home/hopkilau/_tile2vec/"
drive = "/datadrive/tile2vec/"


# for process_data.py
data_dir = home_dir + "data/"
#tif_dir =  data_dir + "tif/unprocessed_naip/"  # earth engine exports
tif_path = drive + "/data/tif/synthetic_naip_10m_50k_oregon/"
tif_dir_naip = drive + "data/tif/unprocessed_naip_10m/"
tif_dir_nlcd = drive + "data/tif/synthetic_5k_nlcd/"  # "data/tif/unprocessed_nlcd_10m/"
np_path = drive + "data/npy/synthetic_naip_10m_50k_oregon/" # output dir for process_data  
np_dir_naip = drive + "data/npy/oregon_naip_10mTEST/"
tif_dir_nlcd = drive + "data/npy/synthetic_nlcd_5k_oregon_275x201/"
np_dir_nlcd = drive + "data/npy/synthetic_nlcd_5k_oregon_200x200/"

# for make_triplets.py
#all_images = data_dir + "npy/all/" # contains all images
all_images = drive + "data/npy_10m/all/"
train_images = drive + "data/npy/train_naip/" # contains train images
test_images = drive + "data/npy/test_mini/" # contains test images
train_images_nlcd = drive + "data/npy/train_nlcd/"
test_images_nlcd = drive + "data/npy/test_nlcd/"
#ebird_tifs = data_dir + "tif/ebird_naip/" # contains images centered at ebird records
ebird_tifs = data_dir + "tif/synthetic_5k_naip_mini/"  # contains images centered at randomly sampled points

# for make_triplets_synthetic.py
synthetic_tiles = drive + "data/npy/synthetic_naip_10m_50k_oregon/"  # "data/npy/synthetic_5k_naip_20m_clipped/"
synthetic_triplets_train = drive + "data/npy/50k/synthetic_10m_50k_train/"  # "data/npy/synthetic_train_50k/"
synthetic_triplets_test = drive + "data/npy/50k/synthetic_10m_50k_test/"  # "data/npy/synthetic_test_50k/"
synthetic_triplets_val = drive + "data/npy/50k/synthetic_10m_50k_val/"  # "data/npy/synthetic_val_50k/"
synthetic_triplets_train_duplicates = drive + "data/npy/15k/synthetic_10m_15k_train_duplicates/"  # "data/npy/synthetic_train_50k/"
synthetic_triplets_test_duplicates = drive + "data/npy/15k/synthetic_10m_15k_test_duplicates/"  # "data/npy/synthetic_test_50k/"
synthetic_triplets_val_duplicates = drive + "data/npy/15k/synthetic_10m_15k_val_duplicates/"  # "data/npy/synthetic_val_50k/"


# triplets for training/val
#train_tiles = drive + "data/tiles_20m/train/"
train_tiles = synthetic_triplets_train # for synthetic sdm
#val_tiles = drive + "data/tiles_20m/val/"
val_tiles = synthetic_triplets_val
#test_tiles = drive + "data/tiles_20m/test/"
test_tiles = synthetic_triplets_test

train_tiles_nlcd = drive + "data/tiles_10m_smart/train_nlcdTEST/"
val_tiles_nlcd = drive + "data/tiles_10m_smart/val_nlcd/"
test_tiles_nlcd = drive + "data/tiles_10m_smart/test_nlcd/"
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
ebird_tifs = drive + "data/tif/synthetic_5k_naip_10m/"
synthetic_naip = drive + "data/npy/synthetic_naip_10m_50k_oregon/"  # "data/npy/synthetic_5k_naip_20m_clipped/"

tifs_to_extract = synthetic_naip

#resizing
to_resize = tif_dir_nlcd
resized = drive + "data/npy/synthetic_nlcd_5k_oregon_200x200/"

