import numpy as np
from sklearn import datasets

import data
import utils
import learn
import spn

def test_partition_net():
  print('Loading dataset...')
  raw_data = datasets.load_breast_cancer()

  print('Reshaping and extracting info...')
  D = np.column_stack((raw_data.data, raw_data.target))
  S = [*range(D.shape[1])]

  print('Generating partition graph...')
  # G = utils.PartitionGraph(D, S)
  G = utils.PartitionNetwork(D, S)

  print('Computing min-cut...')
  P1, P2 = G.partition()

  print('P1:', P1)
  print('---')
  print('P2:', P2)

def fetch_sklearn_data():
  raw_data = datasets.load_breast_cancer()
  D = np.column_stack((raw_data.data, raw_data.target))
  S = [*range(D.shape[1])]
  return D, S

def fetch_data(name: str):
  R, V, T = data.load(name)
  S = [*range(R.shape[1])]
  return R, V, T, S

def learn_structure(D, Sc):
  S = learn.Learn(D, Sc, 2)
  spn.save(S, 'spn.net')
  return S

_, _, D, Sc = fetch_data('accidents')
S = learn_structure(D, Sc)
