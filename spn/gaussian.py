import scipy.stats

from .leaf import Leaf
from .node import Node, VarSet

class Gaussian(Leaf):
  def __init__(self, v: int, mean: float = 0.0, std: float = 1.0, id: int = None):
    Leaf.__init__(self, id)
    self._dist = scipy.stats.norm()
    self._mean = mean
    self._std = std
    self._v = v

  def add(self, c: Node):
    raise ValueError('Leaf nodes cannot have children!')

  def eval(self, e: VarSet) -> float:
    for x, v in enumerate(e):
      if x == self._v:
        return self._dist.logpdf(v, loc=self._mean, scale=self._std)
    return 0.0

  def distribution(self):
    return 'gaussian'
