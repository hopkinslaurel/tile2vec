#! /usr/bin/Rscript

# To be used with make_triplets.py --smart
# make_triplets.py --smart creates X potential 'neighbor' and X potential 'distant' tiles.
# This script then selects the neighbor and distant tiles based on majority NLCD label and 
# Fragstats metrics and deletes the remaining potential tiles

library(raster)
#library(rgdal)
library(landscapemetrics)
library(reticulate)

np <- import("numpy")

path <- "C:\\Users\\Laurel\\Documents\\Oregon State\\Research\\oregon_2011\\Data\\tile2vec\\tiles_10m_smart\\test_nlcd\\"

npy_files <- list.files(path=path, pattern="*.npy", full.names=TRUE, recursive=FALSE)
#npy_files <- list.files(path=".", pattern="*.npy", full.names=TRUE, recursive=FALSE)

#npy_files


# --- for several files --- # 
# 0_anchor.npy. 0_a_neighbor.npy, 0_b_neighbor.npy, ..., 0_a_distant.npy, 0_b_distant.npy, ...  
metrics <- lapply(npy_files, function(x) {
  img_name <- basename(x)
  #img_name <- tail(img_name,n=2)[1]  # use when name end with _clipped
  names <- unlist(strsplit(unlist(strsplit(img_name, ".npy")[1]), "_"))
  if (length(names) == 4) {
    img_number <- names[1]
    index <- names[2]
    tile <- names[3]
  } else {
    img_number <- names[1]
    index <- NA
    tile <- names[2]
  }
  
  print(img_number)
  
  nlcd_matrix <- np$load(x)
  nlcd_raster <- raster(nlcd_matrix)
  
  # claculate majority class
  majority_class <- strtoi(names(which.max(table(nlcd_matrix))))
  
  # calculate fragstats stat
  cai_mn <- lsm_l_cai_mn(nlcd_raster, consider_boundary = FALSE)[["value"]]
  
  data.frame (img_number=img_number, index=index, tile=tile, majority_nlcd = majority_class, l_cai_mn = cai_mn)
  
})

out <- do.call(rbind, metrics)
#write.csv(frag_metrics, file=paste(path, "fragstats_metrics.csv", sep=""), row.names=FALSE)

# now, find best match: 

# TEST DATA 
img_number <- c(0,0,0,0,0,0,0)
version <- c(NA, 'a', 'b', 'c', 'a', 'b', 'c')
tile <- c('anchor','neighbor','neighbor','neighbor','distant','distant','distant')
majority_nlcd <- c(42,42,42,41,52,31,42)
l_cai_mn <- c(35,51,37,20,19,20,80)
test_out <- data.frame(img_number,version,tile,majority_nlcd,l_cai_mn)

test_out <- out 

###NEED TO PUT THIS IN A LOOP, DO EACH img_number SEPARATELY 
set = 1
# --- compare anchor to potential neighbor and distant tiles --- #
neighbor_df <- test_out[which(test_out$tile=='neighbor' & test_out$img_number==set), ]
distant_df <- test_out[which(test_out$tile=='distant' & test_out$img_number==set), ]

# 1) must have same majority nlcd class
majority_class <- test_out$majority_nlcd[test_out$tile=='anchor' & test_out$img_number==set] # get majority class
filtered_neighbor_df <- neighbor_df[which(neighbor_df$majority_nlcd==majority_class), ]  
# if there are no other tiles with the same majority landscape, don't filter by nlcd class
if (nrow(filtered_neighbor_df) == 0) {
  filtered_neighbor_df <- neighbor_df
}

# 2) must have most similar fragstats metrics
frag_metric <- test_out$l_cai_mn[test_out$tile=='anchor'& test_out$img_number==set] # get majority class
filtered_neighbor_df$cai_diff <- abs(filtered_neighbor_df$l_cai_mn - frag_metric)
if (nrow(filtered_neighbor_df) == 1) {
  keep <- filtered_neighbor_df$index
} else {
  keep <- filtered_neighbor_df$index[min(filtered_neighbor_df$cai_diff)]
}
 
## delete all other files


## repeat for distant tiles
# 1) must have same majority nlcd class (or do we want different classes?)


# delete remaining files
#dirname(X)
#file.remove("some_other_file.csv")


# --- for a single file --- #
npy <- npy_files[1]

nlcd_matrix <- np$load(npy)
nlcd_raster <- raster(nlcd_matrix)
#nlcd_raster
#plot(nlcd_raster)

# find majority class from matrix
#majority_class <- strtoi(names(which.max(table(nlcd_matrix))))  # names get name from named int

# calculate fragstats stat
#cai_mn <- lsm_l_cai_mn(nlcd_raster, consider_boundary = FALSE)[["value"]]

