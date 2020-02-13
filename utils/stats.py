import math

import numpy as np

import spn

def mean(D: spn.Dataset, p: tuple = None):
  if p is not None and np.prod(D.shape) != len(D):
    D = D[:,p]
  return D.mean()

def std(D: spn.Dataset, p: tuple = None):
  if p is not None and np.prod(D.shape) != len(D):
    D = D[:,p]
  return D.std()

def logprobs(D: spn.Dataset, p: tuple = None):
  if D.size == 1:
    l = np.full(2, -math.inf)
    l[int(D[0])] = 0
    return l
  if len(D.shape) != 1 and p is not None:
    D = D[:,p]
  l = np.log(np.bincount(D.astype(int), minlength=2)/D.size)
  return l

# split_by indices
def split_by(D: spn.Dataset, I: list):
  n = D[0].shape[0]
  L = [np.empty((0, n))] * (np.max(I)+1)
  for i, l in enumerate(I):
    L[l] = np.vstack((L[l], D[i]))
  return L

# Returns a dataset containing the columns in P, where P is a subset of S.
def restrict(D: spn.Dataset, S: spn.Scope, P: spn.Scope):
  _, I, _ = np.intersect1d(S, P, assume_unique = True, return_indices = True)
  return D[:,I]

# Groups elements indices by their values. If they have the same value, group them together. This
# effectively returns a partition of where each element is in each set.
def group_by(L):
  I = np.argsort(L)
  S = L[I]
  return np.split(I, np.cumsum(np.diff(np.nonzero(np.concatenate(([True], S[1:] != S[:-1])))[0])))

# D: dataset
# x: variable, st x <- x - min_x Index(Variables)
# y: variable, st y <- y - min_y Index(Variables)
# If scope is always in increasing order, then x and y are exactly the vertices from VertexMap
def correlation(D: spn.Dataset, x: int, y: int) -> float:
  X = D[:,x]
  Y = D[:,y]

  x_mean = X.mean()
  y_mean = Y.mean()
  X_diff = X-x_mean
  Y_diff = Y-y_mean
  top = np.multiply(X_diff, Y_diff).sum()

  x_sq = math.sqrt(np.multiply(X_diff, X_diff).sum())
  y_sq = math.sqrt(np.multiply(Y_diff, Y_diff).sum())
  bottom = x_sq*y_sq

  if top == 0 and bottom == top:
    return 0.0
  return abs(top/bottom)
