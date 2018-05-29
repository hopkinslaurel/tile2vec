import numpy as np
import pandas as pd
import random
from scipy import stats
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn import cross_validation, linear_model
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.collections import EllipseCollection
import seaborn as sns
import pdb
from fig_utils import *
import paths
import pickle

def overlay(country, y1, y1_hat, r2_1, y2, y2_hat, r2_2,  margin, overlay=False):
    """
    Plots consumption predictions vs. true values.
    """
    slope1, intercept1, y1min, y1max, x1min, x1max = compute_plot_params(
        y1, y1_hat, margin)
    slope2, intercept2, y2min, y2max, x2min, x2max = compute_plot_params(
        y2, y2_hat, margin)
    sns.set_style('white')
    plt.figure()
    plt.axis('equal')
    plt.scatter(y1, y1_hat, edgecolor='k', color='lightblue', s=20, marker='o', alpha=.5)
    plt.scatter(y2, y2_hat, edgecolor='k', color='red', s=20, marker='o', alpha=1)
    x_trend = np.array([x1min, x1max]) * 1.5
    y_trend = slope1 * x_trend + intercept1
    plt.plot(x_trend, y_trend, 'b-', linewidth=2,
             color=sns.xkcd_rgb['french blue'], alpha=.5)
    x_trend = np.array([x2min, x2max]) * 1.5
    y_trend = slope2 * x_trend + intercept2
    plt.plot(x_trend, y_trend, 'b-', linewidth=2,
             color=sns.xkcd_rgb['pale red'], alpha=1)

    plt.xlim(min(x1min,x2min), max(x1max,x2max))
    plt.ylim(min(y1min,y2min), max(y1max,y2max))
    plt.xlabel('Log consumption expenditures', fontsize=14)
    plt.ylabel('Model predictions', fontsize=14)
    plt.title(country.capitalize() + ': $r^2 = {0:.3f}$'.format(r2_1))
    plt.show()
    plt.savefig('overlay.png')
    plt.close("all")

country = 'uganda'
country_path = paths.original_data
dimension = None
k = 5
k_inner = 5
points = 10
alpha_low = 1
alpha_high = 5
regression_margin = 0.25

X, y1, y1_hat, r2_1, mse = predict_consumption(country, country_path,
                                               dimension, k, k_inner,
                                               points, alpha_low,
                                               alpha_high,
                                               regression_margin)
print(r2_1)
with open('models/practice/y_small_e50.p','rb') as f:
    y2, y2_hat, r2_2 = pickle.load(f)
print(r2_2)
overlay(country, y1, y1_hat, r2_1, y2, y2_hat, r2_2, regression_margin)
