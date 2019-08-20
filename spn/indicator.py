import math

from .leaf import Leaf
from .node import VarSet

class Indicator(Leaf):
  def __init__(self, var: int, val: int, id: int = None):
    Leaf.__init__(self, id)
    self._v = var
    self._x = val

  def eval(self, e: VarSet) -> float:
    for x, v in enumerate(e):
      if x == self._v:
        if abs(self._x - v) < 1e-6:
          return 0.0
        else:
          return -math.inf
    return 0.0

  def distribution(self):
    return 'indicator'
