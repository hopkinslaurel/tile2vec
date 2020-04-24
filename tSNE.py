import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
from matplotlib.ticker import NullFormatter 
from time import time


# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--species', dest='species')
parser.add_argument('--model', dest='model')
parser.add_argument('--seed', dest='seed', type=int, default=0)
args = parser.parse_args()

# perplexities for t-SNE
perplexities = [5, 30, 50, 100]

species = 'response_'+args.species
species_responses = pd.read_csv('OR_2011_synthetic_responses.csv', header=0, index_col=0)
features = pd.read_csv('features/OR_2011_synthetic_' + args.model + '_features.csv', header=None, index_col=0)

df_all = pd.merge(features,species_responses[[species]], how='left', left_index=True, right_index=True)
red = (df_all[species] == 0).values
blue = (df_all[species] == 1).values

(fig, subplots) = plt.subplots(1, len(perplexities), figsize=(15, 8))
ax = subplots[0]#[0]
ax.xaxis.set_major_formatter(NullFormatter())
ax.yaxis.set_major_formatter(NullFormatter())
plt.axis('tight')

for i, perplexity in enumerate(perplexities):
    ax = subplots[i]

    t0 = time()
    tsne = TSNE(n_components=2, init='random', random_state=args.seed, perplexity=perplexity) # potential put this in a loop with different perlexities
    Y = tsne.fit_transform(df_all.drop([species], axis=1).values)
    t1 = time()
    print("t-SNE, perplexity=%d in %.2g sec" % (perplexity, t1 - t0))

    ax.set_title("Perplexity=%d" % perplexity)
    ax.scatter(Y[red, 0], Y[red, 1], c="r")
    ax.scatter(Y[blue, 0], Y[blue, 1], c="b")
    ax.xaxis.set_major_formatter(NullFormatter())
    ax.yaxis.set_major_formatter(NullFormatter())
    ax.axis('tight')

fig.suptitle('Species: {}, Features: {}'.format(args.species, args.model))
plt.show()
