import math

from sklearn.cluster import KMeans

import spn
import utils

def __factor(D: spn.Dataset, S: spn.Scope, k: int, dist: str = "multinomial"):
  pi = spn.Product()
  for i in range(0, D.shape[1]):
    pi.add(LearnMST(D[:,i], [S[i]], k, dist))
  return pi

def LearnKL(D: spn.Dataset, S: spn.Scope, k: int, dist: str = 'multinomial'):
  if len(S) == 1:
    print('Creating univariate distribution...')
    if dist == 'multinomial':
      print('  ... created Multinomial.')
      return spn.Multinomial(S[0], utils.logprobs(D, 0))
    else:
      print('  ... created Gaussian.')
      return spn.Gaussian(S[0], utils.mean(D, S), utils.std(D, S))
  print('Clustering...')
  if D.shape[0] < k:
    print(D)
    pi = spn.Product()
    for i in range(0, D.shape[1]):
      print(' ', i, D[:,i], [S[i]])
      pi.add(LearnKL(D[:,i], [S[i]], k, dist))
    return pi
  s = spn.Sum()
  C = utils.split_by(D, KMeans(n_clusters = k).fit(D).labels_)
  if len(C) == 1:
    v = C[0]
    n = len(v)//2
    C = [v[:n], v[n:]]
    print(C[0].shape, C[1].shape)
  for c in C:
    print('{}/{}={}'.format(c.size, D.size, c.size/D.size))
    m = c.size/D.size
    print('Partitioning...')
    N = utils.PartitionNetwork(c, S)
    P = N.partition()
    pi = spn.Product()
    s.add(pi, math.log(m))
    for p in P:
      pi.add(LearnKL(utils.restrict(c, S, p[0]), p[0], k, dist))
      pi.add(LearnKL(utils.restrict(c, S, p[1]), p[1], k, dist))
  return s

def LearnMST(D: spn.Dataset, S: spn.Scope, k: int, dist: str = 'multinomial'):
  print('|S| = {}, S = {}'.format(len(S), S))
  if len(S) == 1:
    print('Creating univariate distribution...')
    if dist == 'multinomial':
      print('  ... created Multinomial.')
      return spn.Multinomial(S[0], utils.logprobs(D, 0))
    else:
      print('  ... created Gaussian.')
      return spn.Gaussian(S[0], utils.mean(D, S), utils.std(D, S))
  print('Clustering...')
  if D.shape[0] < k:
    return __factor(D, S, k, dist)
  C = utils.split_by(D, KMeans(n_clusters = k).fit(D).labels_)
  s = spn.Sum()
  M = {}
  for c in C:
    m = c.size/D.size
    print('Partitioning...')
    N = utils.PartitionGraph(c, S)
    P = N.partition_mst()
    for p in P:
      pi = spn.Product()
      s.add(pi, math.log(m))
      for q in p:
        t = tuple(q)
        if t not in M:
          ch = LearnMST(utils.restrict(c, S, q), q, k, dist)
          M[t] = ch
        else:
          print('Reuse: {}'.format(t))
        pi.add(M[t])
  return s

Learn = LearnMST
