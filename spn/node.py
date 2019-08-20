from __future__ import annotations
from typing import Dict, List

import numpy as np

from . import *

VarSet = Dict[int, float]

Dataset = np.ndarray

Scope = List[int]

_global_id_counter = 0

class Node:
  def __init__(self, id: int = None):
    self._ch = []
    if id is None:
      global _global_id_counter
      id = _global_id_counter
      _global_id_counter += 1
    self._id = id

  def add(self, c: 'Node'):
    self._ch.append(c)

  def eval(self, e: VarSet) -> float:
    raise NotImplementedError

  def top_down_order(self) -> List[Node]:
    O = []
    Q = [self]
    V = {}
    while len(Q) > 0:
      n = Q.pop(0)
      O.append(n)
      for c in n._ch:
        if c not in V:
          V[c] = True
          Q.append(c)
    return O