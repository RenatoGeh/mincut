import math

import numpy as np

import spn

def mean(D: spn.Dataset, p: tuple = None):
  if tuple is not None:
    D = D[:,p]
  return D.mean()

def std(D: spn.Dataset, p: tuple = None):
  if tuple is not None:
    D = D[:,p]
  return D.std()

def logprobs(D: spn.Dataset, p: tuple = None):
  if D.size == 1:
    l = np.full(2, -math.inf)
    l[int(D[0])] = 0
    return l
  if tuple is not None:
    D = D[:,p]
  l = np.log(np.bincount(D.astype(int), minlength=2)/D.size)
  print(l)
  return l

# split_by indices
def split_by(D: spn.Dataset, I: list):
  n = D[0].shape[0]
  L = [np.empty((0, n))] * (np.max(I)+1)
  for i, l in enumerate(I):
    L[l] = np.vstack((L[l], D[i]))
  return L

def restrict(D: spn.Dataset, S: spn.Scope, P: spn.Scope):
  _, I, _ = np.intersect1d(S, P, assume_unique = True, return_indices = True)
  return D[:,I]

# Groups elements indices by their values. If they have the same value, group them together. This
# effectively returns a partition of where each element is in each set.
def group_by(L):
  I = np.argsort(L)
  S = L[I]
  return np.split(I, np.cumsum(np.diff(np.nonzero(np.concatenate(([True], S[1:] != S[:-1])))[0])))
