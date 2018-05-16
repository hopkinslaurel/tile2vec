# Neal Jean/Figure 3 notebook (see predicting-poverty repo)
# took things I needed
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

def predict_consumption(
    country, country_path, dimension=None, k=5, k_inner=5, points=10,
        alpha_low=1, alpha_high=5, margin=0.25):
    """
    Plots predicted consumption
    """
    X_full, _, y = load_country_lsms(country_path)
    X = reduce_dimension(X_full, dimension)
    y_hat, r2, mse = run_cv(X, y, k, k_inner, points, alpha_low, alpha_high)
    plot_predictions(country, y, y_hat, r2, margin)
    return X, y, y_hat, r2, mse


def plot_predictions(country, y, y_hat, r2, margin):
    """
    Plots consumption predictions vs. true values.
    """
    slope, intercept, ymin, ymax, xmin, xmax = compute_plot_params(
        y, y_hat, margin)
    sns.set_style('white')
    plt.figure()
    plt.axis('equal')
    plt.scatter(y, y_hat, edgecolor='k', color='lightblue', s=20, marker='o')
    x_trend = np.array([xmin, xmax]) * 1.5
    y_trend = slope * x_trend + intercept
    plt.plot(x_trend, y_trend, 'b-', linewidth=2,
             color=sns.xkcd_rgb['french blue'])
    plt.xlim(xmin, xmax)
    plt.ylim(ymin, ymax)
    plt.xlabel('Log consumption expenditures', fontsize=14)
    plt.ylabel('Model predictions', fontsize=14)
    plt.title(country.capitalize() + ': $r^2 = {0:.2f}$'.format(r2))
    plt.show()
    plt.savefig('prediction.png')
    plt.close("all")

def compute_plot_params(y, y_hat, margin):
    """
    Computes parameters for plotting consumption predictions vs. true values.
    """
    slope, intercept, _, _, _ = stats.linregress(y, y_hat)
    ymin = min(y_hat) - margin
    ymax = max(y_hat) + margin
    xmin = min(y) - margin
    xmax = max(y) + margin
    return slope, intercept, ymin, ymax, xmin, xmax


def compute_r2(model, X, y):
    """
    Computes model r2.
    """
    y_hat = model.predict(X)
    r2 = stats.pearsonr(y, y_hat)[0] ** 2
    return r2

############################
######### General ##########
############################


def load_country_lsms(country_path):
    """
    Loads data for one LSMS country.
    """
    X = np.load(country_path + 'cluster_conv_features.npy')
    X_nl = np.load(country_path + 'cluster_nightlights.npy').reshape(-1, 1)
    y = np.load(country_path + 'cluster_consumptions.npy')
    hhs = np.load(country_path + 'cluster_households.npy')
    images = np.load(country_path + 'cluster_image_counts.npy')
    # Filter out single households and <10 images
    mask = np.logical_and((hhs >= 2), (images >= 10))
    X = X[mask]
    X_nl = X_nl[mask]
    y = y[mask]
    # Filter out 0 consumption clusters
    X = X[y > 0]
    X_nl = X_nl[y > 0]
    y = y[y > 0]
    y = np.log(y)
    return X, X_nl, y


def load_country_dhs(country_path):
    """
    Loads data for one DHS country.
    """
    X = np.load(country_path + 'cluster_conv_features.npy')
    X_nl = np.load(country_path + 'cluster_nightlights.npy').reshape(-1, 1)
    y = np.load(country_path + 'cluster_assets.npy')
    hhs = np.load(country_path + 'cluster_households.npy')
    images = np.load(country_path + 'cluster_image_counts.npy')
    # Filter out single households and <10 images
    mask = np.logical_and((hhs >= 2), (images >= 10))
    X = X[mask]
    X_nl = X_nl[mask]
    y = y[mask]
    return X, X_nl, y


def reduce_dimension(X, dimension):
    """
    Uses PCA to reduce dimensionality of features.
    """
    if dimension is not None:
        pca = PCA(n_components=dimension)
        X = pca.fit_transform(X)
    return X


def load_and_reduce_country(country_path, survey, dimension):
    """
    Loads data for one country and reduces the dimensionality of features.
    """
    if survey == 'lsms':
        X, _, y = load_country_lsms(country_path)
    elif survey == 'dhs':
        X, _, y = load_country_dhs(country_path)
    X = reduce_dimension(X, dimension)
    return X, y


def run_cv(X, y, k, k_inner, points, alpha_low, alpha_high, randomize=False):
    """
    Runs nested cross-validation to make predictions and compute r-squared.
    """
    alphas = np.logspace(alpha_low, alpha_high, points)
    r2s = np.zeros((k,))
    mses = np.zeros((k,))
    y_hat = np.zeros_like(y)
    kf = cross_validation.KFold(n=y.size, n_folds=k, shuffle=True)
    fold = 0
    for train_idx, test_idx in kf:
        r2s, mses, y_hat, fold = evaluate_fold(
            X, y, train_idx, test_idx, k_inner, alphas, r2s, mses, y_hat, fold,
            randomize)
    return y_hat, r2s.mean(), mses.mean()


def scale_features(X_train, X_test):
    """
    Scales features using StandardScaler.
    """
    X_scaler = StandardScaler(with_mean=True, with_std=False)
    X_train = X_scaler.fit_transform(X_train)
    X_test = X_scaler.transform(X_test)
    return X_train, X_test


def train_and_predict_ridge(alpha, X_train, y_train, X_test):
    """
    Trains ridge model and predicts test set.
    """
    ridge = linear_model.Ridge(alpha)
    ridge.fit(X_train, y_train)
    y_hat = ridge.predict(X_test)
    return y_hat


def predict_inner_test_fold(X, y, y_hat, train_idx, test_idx, alpha):
    """
    Predicts inner test fold.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)
    y_hat[test_idx] = train_and_predict_ridge(alpha, X_train, y_train, X_test)
    return y_hat


def find_best_alpha(X, y, k_inner, alphas):
    """
    Finds the best alpha in an inner CV loop.
    """
    kf = cross_validation.KFold(n=y.size, n_folds=k_inner, shuffle=True)
    best_alpha = 0
    best_r2 = 0
    for idx, alpha in enumerate(alphas):
        y_hat = np.zeros_like(y)
        for train_idx, test_idx in kf:
            y_hat = predict_inner_test_fold(
                X, y, y_hat, train_idx, test_idx, alpha)
        r2 = stats.pearsonr(y, y_hat)[0] ** 2
        if r2 > best_r2:
            best_alpha = alpha
            best_r2 = r2
    return best_alpha


def evaluate_fold(
        X, y, train_idx, test_idx, k_inner, alphas, r2s, mses, y_hat, fold,
        randomize):
    """
    Evaluates one fold of outer CV.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    if randomize:
        random.shuffle(y_train)
    best_alpha = find_best_alpha(X_train, y_train, k_inner, alphas)
    X_train, X_test = scale_features(X_train, X_test)
    y_test_hat = train_and_predict_ridge(best_alpha, X_train, y_train, X_test)
    r2 = stats.pearsonr(y_test, y_test_hat)[0] ** 2
    mse = ((y_test - y_test_hat) ** 2).mean()
    r2s[fold] = r2
    mses[fold] = mse
    y_hat[test_idx] = y_test_hat
    return r2s, mses, y_hat, fold + 1
