# t-SNE in PyTorch
This repository contains a implementation of [t-Distributed Stochastic Neighbor Embedding (t-SNE)](https://lvdmaaten.github.io/tsne/) in PyTorch.

## Files
- `train.py` is a training file for t-SNE. Note that data has to be
preprocessed.
- `preprocess.py` preprocesses MNIST data for use in the training file. The
preprocessing function can be imported directly for use.
- `tsne_module.py` a t-SNE PyTorch file.

## Resources
The code is based off of the original paper as well as some other
implementations of t-SNE
- [scikit-learn t-SNE](https://github.com/scikit-learn/scikit-learn/blob/fd237278e895b42abe8d8d09105cbb82dc2cbba7/sklearn/manifold/_t_sne.py)