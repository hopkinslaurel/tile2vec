import numpy as np
import pandas as pd
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
import argparse
from sklearn import datasets
from matplotlib.ticker import NullFormatter 
from time import time
from mpl_toolkits import mplot3d


# Parse command line arguments
parser = argparse.ArgumentParser()

parser.add_argument('--species', dest='species')
parser.add_argument('--model', dest='model')
parser.add_argument('--perplexity', dest='perplexity', type=int, default=30)
parser.add_argument('--seed', dest='seed', type=int, default=0)
args = parser.parse_args()

species = 'response_'+args.species
#species = args.species
#species_responses = pd.read_csv('OR_2011_synthetic_responses.csv', header=0, index_col=0)
species_responses = pd.read_csv('OR_2011_synthetic_response_with_nlcd.csv', header=0, index_col=0)
features = pd.read_csv('~/features/OR_2011_synthetic_' + args.model + '_features.csv', header=None, index_col=0)

df_all = pd.merge(features,species_responses[[species]], how='left', left_index=True, right_index=True)
red = (df_all[species] == 0).values
blue = (df_all[species] == 1).values

fig = plt.figure()
ax = plt.axes(projection='3d')

t0 = time()
tsne = TSNE(n_components=3, init='random', random_state=args.seed, perplexity=args.perplexity) # potential put this in a loop with different perlexities
Y = tsne.fit_transform(df_all.drop([species], axis=1).values)
t1 = time()
print("t-SNE, perplexity=%d in %.2g sec" % (args.perplexity, t1 - t0))
print(Y)

ax.set_title("Perplexity=%d" % args.perplexity)
ax.scatter3D(Y[red, 0], Y[red, 1], Y[red, 2], c="r")
ax.scatter3D(Y[blue, 0], Y[blue, 1], Y[blue, 2], c="b")
    
fig.suptitle('Species: {}, Features: {}'.format(args.species, args.model))
#fig.suptitle('Majority Forest, Features: {}'.format(args.model))
plt.show()
