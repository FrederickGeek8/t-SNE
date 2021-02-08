import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)

from sklearn.metrics.pairwise import pairwise_distances
from sklearn.manifold.t_sne import _joint_probabilities
from scipy.spatial.distance import squareform
from sklearn import datasets
import pickle
import argparse
import numpy as np


def preprocess(x, metric='euclidean', perplexity=30):
    dist = pairwise_distances(x, metric=metric, squared=True)
    p = _joint_probabilities(dist, perplexity, 0)

    return p


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-out", "--out_file", type=str, required=True)
    args = parser.parse_args()

    digits = datasets.load_digits()
    pij = preprocess(digits.data)

    f = open(f"{args.out_file}", "wb")
    np.save(f, pij)