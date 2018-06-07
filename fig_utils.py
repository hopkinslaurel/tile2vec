# fig_utils: predicting-poverty repo (author: Neal Jean)
# copy pasted from fig_utils, removing unneeded parts
# To re-create figure 4A of science paper, I added
# Tile2Vec curve and manually put in PCA dimension of 10
# for all X_tf data. 
# I am getting runtime warning with scipy stats when running this
# w/ poverty_plot "invalid value encountered"

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

############################
######### Figure 3 #########
############################

def predict_consumption(
    country, country_path, dimension=None, k=5, k_inner=5, points=10,
        alpha_low=1, alpha_high=5, margin=0.25, exp=None):
    """
    Plots predicted consumption
    """
    X_full, _, _, y = load_country_lsms(country_path, exp=exp)
    X = reduce_dimension(X_full, dimension)
    y_hat, r2, mse = run_cv(X, y, k, k_inner, points, alpha_low, alpha_high)
    plot_predictions(country, y, y_hat, r2, margin)
    return X, y, y_hat, r2, mse


def plot_predictions(country, y, y_hat, r2, margin, exp=None):
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
    if exp is not None:
        plt.savefig('prediction_' + exp + '.png')
    else:
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

############################
######## Figure 4ab ########
############################


def compare_models(
        country_path, survey, percentiles, dimension, k, k_inner, points, alpha_low, alpha_high, trials,
        poverty_line, multiples, exp=None):
    """
    Evaluates and plots comparison of transfer learning and nightlight models.
    """
    r2s, r2s_nl, r2s_tf = evaluate_percentiles(
        country_path, survey, percentiles, dimension, k, k_inner, points, alpha_low, alpha_high, trials, exp=exp)
    if survey == 'lsms':
        X, X_nl, X_tf, y = load_and_reduce_country_by_percentile(
            country_path, survey, 1.0, dimension, exp=exp)
        fractions = compute_fractions(poverty_line, multiples, y)
        plot_percentiles_lsms(percentiles, multiples, r2s, r2s_nl, r2s_tf, fractions)

def load_and_reduce_country_by_percentile(
        country_path, survey, percentile, dimension, exp=None):
    """
    Loads data for one country up to a certain percentile.
    """
    if survey == 'lsms':
        X, X_nl, X_tf, y = load_country_lsms(country_path, exp=exp)
    X, X_nl, X_tf, y = threshold_by_percentile(X, X_nl, X_tf, y, percentile)
    X = reduce_dimension(X, dimension)
    X_tf = reduce_dimension(X, 10) #adding this extra step
    return X, X_nl, X_tf, y


def threshold_by_percentile(X, X_nl, X_tf, y, percentile):
    """
    Threshold data by output percentile.
    """
    threshold = np.percentile(y, q=100*percentile)
    X = X[y <= threshold]
    X_nl = X_nl[y <= threshold]
    X_tf = X_tf[y <= threshold]
    y = y[y <= threshold]
    return X, X_nl, X_tf, y


def evaluate_percentiles(
        country_path, survey, percentiles, dimension, k, k_inner, points, alpha_low, alpha_high, trials, exp=None):
    """
    Evaluate transfer learning and nightlight models for each percentile.
    """
    r2s = np.zeros((len(percentiles), trials))
    r2s_nl = np.zeros((len(percentiles), trials))
    r2s_tf = np.zeros((len(percentiles), trials))
    for idx, percentile in enumerate(percentiles):
        for trial in range(trials):
            X, X_nl, X_tf, y = load_and_reduce_country_by_percentile(
                country_path, survey, percentile, dimension, exp=exp)
            _, r2, _ = run_cv(
                X, y, k, k_inner, points, alpha_low, alpha_high,
                randomize=False)
            _, r2_tf, _ = run_cv(
                X_tf, y, k, k_inner, points, alpha_low, alpha_high,
                randomize=False)
            r2_nl = run_cv_ols(X_nl, y, k)
            r2s[idx, trial] = r2
            r2s_nl[idx, trial] = r2_nl
            r2s_tf[idx, trial] = r2_tf
    r2s = r2s.mean(axis=1)
    r2s_nl = r2s_nl.mean(axis=1)
    r2s_tf = r2s_tf.mean(axis=1)
    return r2s, r2s_nl, r2s_tf


def run_cv_ols(X, y, k):
    """
    Runs OLS in cross-validation to compute r-squared.
    """
    r2s = np.zeros((k,))
    kf = cross_validation.KFold(n=y.size, n_folds=k, shuffle=True)
    fold = 0
    for train_idx, test_idx in kf:
        r2s, fold = evaluate_fold_ols(X, y, train_idx, test_idx, r2s, fold)
    return r2s.mean()


def evaluate_fold_ols(X, y, train_idx, test_idx, r2s, fold):
    """
    Evaluates one fold of outer CV using OLS.
    """
    X_train, X_test = X[train_idx], X[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    X_train, X_test = scale_features(X_train, X_test)
    y_test_hat = train_and_predict_ols(X_train, y_train, X_test)
    r2 = stats.pearsonr(y_test, y_test_hat)[0] ** 2
    if np.isnan(r2):
        r2 = 0
    r2s[fold] = r2
    return r2s, fold + 1


def train_and_predict_ols(X_train, y_train, X_test):
    """
    Trains OLS model and predicts test set.
    """
    ols = linear_model.LinearRegression()
    ols.fit(X_train, y_train)
    y_hat = ols.predict(X_test)
    return y_hat


def compute_fractions(poverty_line, multiples, y):
    """
    Computes the fraction of clusters below each multiple of the poverty line.
    """
    fractions = np.zeros((len(multiples),))
    for idx, multiple in enumerate(multiples):
        fractions[idx] = (
            np.exp(y) <= poverty_line * multiple).sum() / float(y.size)
    return fractions


def plot_percentiles_lsms(percentiles, multiples, r2s, r2s_nl, r2s_tf, fractions):
    """
    Plots transfer learning model vs. nightlights model at each percentile.
    """
    sns.set_style('white')
    plt.figure(figsize=(6, 6))
    lines = []
    percentiles = [100 * x for x in percentiles]
    for idx, multiple in enumerate(multiples):
        lines.append(
            plt.axvline(
                100 * fractions[idx], color='r', linestyle='dashed',
                linewidth=3.0 / (idx + 1),
                label=str(multiple) + 'x poverty line'))
    line_legend = plt.legend(
        handles=lines, title='Poverty line multiples:', loc='upper right',
        bbox_to_anchor=(0.5, 1), fontsize=10)
    plt.gca().add_artist(line_legend)
    #curve1, = plt.plot(percentiles, r2s, label='Transfer learning')
    #curve2, = plt.plot(percentiles, r2s_nl, label='Nightlights')
    curve1, = plt.plot(percentiles, r2s, label='Tile2Vec')
    curve2, = plt.plot(percentiles, r2s_tf, label='Transfer Learning')
    curve3, = plt.plot(percentiles, r2s_nl, label='Nightlights')
    plt.legend(
        handles=[curve1, curve2, curve3], loc='upper right',
        bbox_to_anchor=(0.5, 0.65))
    plt.xlabel('Poorest percent of clusters used', fontsize=14)
    plt.ylabel('$r^2$', fontsize=18)
    plt.show()
    plt.savefig('poverty_line.png')
    plt.close("all")

############################
######### General ##########
############################


def load_data(country_paths, survey, dimension):
    """
    Loads data for all surveys.
    """
    data = []
    for country_path in country_paths:
        X, y = load_and_reduce_country(country_path, survey, dimension)
        data.append((X, y))
    return data


def load_country_lsms(country_path, exp=None):
    """
    Loads data for one LSMS country.
    """
    if exp is not None:
        X = np.load(country_path + 'cluster_conv_features_' + exp + '.npy')
    else:
        X = np.load(country_path + 'cluster_conv_features.npy')
    X_nl = np.load(country_path + 'cluster_nightlights.npy').reshape(-1, 1)
    X_tf = np.load(country_path + 'cluster_conv_features_tf.npy')
    y = np.load(country_path + 'cluster_consumptions.npy')
    hhs = np.load(country_path + 'cluster_households.npy')
    images = np.load(country_path + 'cluster_image_counts.npy')
    # Filter out single households and <10 images
    mask = np.logical_and((hhs >= 2), (images >= 10))
    X = X[mask]
    X_nl = X_nl[mask]
    X_tf = X_tf[mask]
    y = y[mask]
    # Filter out 0 consumption clusters
    X = X[y > 0]
    X_nl = X_nl[y > 0]
    X_tf = X_tf[y > 0]
    y = y[y > 0]
    y = np.log(y)
    return X, X_nl, X_tf, y

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
