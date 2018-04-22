# Neal Jean/Figure 3 notebook (see predicting-poverty repo)
# minor edits by Anshul Samar

from fig_utils import *
import matplotlib.pyplot as plt
import time

# ## Predicting consumption expeditures
# 
# The parameters needed to produce the plots are as follows:
# 
# - country: Name of country being evaluated as a lower-case string
# - country_path: Path of directory containing LSMS data corresponding to the specified country
# - dimension: Number of dimensions to reduce image features to using PCA. Defaults to None, which represents no dimensionality reduction.
# - k: Number of cross validation folds
# - k_inner: Number of inner cross validation folds for selection of regularization parameter
# - points: Number of regularization parameters to try
# - alpha_low: Log of smallest regularization parameter to try
# - alpha_high: Log of largest regularization parameter to try
# - margin: Adjusts margins of output plot
# 
# The data directory should contain the following 5 files for each country:
# 
# - conv_features.npy: (n, 4096) array containing image features corresponding to n clusters
# - consumptions.npy: (n,) vector containing average cluster consumption expenditures
# - nightlights.npy: (n,) vector containing the average nightlights value for each cluster
# - households.npy: (n,) vector containing the number of households for each cluster
# - image_counts.npy: (n,) vector containing the number of images available for each cluster
# 
# Exact results may differ slightly with each run due to randomly splitting data into training and test sets.

# #### Panel A

# Plot parameters
country = 'uganda'
country_path = '/home/asamar/tile2vec/data/uganda_lsms/'
dimension = None
k = 5
k_inner = 5
points = 10
alpha_low = 1
alpha_high = 5
margin = 0.25

# Plot single panel
t0 = time.time()
X, y, y_hat, r_squareds_test = predict_consumption(country, country_path,
                                dimension, k, k_inner, points, alpha_low,
                                alpha_high, margin)
print(r_squareds_test)
t1 = time.time()
print('Finished in {} seconds'.format(t1-t0))

