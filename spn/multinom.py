import math

from typing import List

from .leaf import Leaf
from .indicator import Indicator
from .sum import Sum
from .node import VarSet

class Multinomial(Leaf):
  def __init__(self, v: int, lpr: List[float], id: int = None):
    Leaf.__init__(self, id)
    self._lpr = lpr
    self._v = v

  def eval(self, e: VarSet) -> float:
    for x, v in enumerate(e):
      if x == self._v:
        if v < 0 and v >= len(self._lpr):
          raise ValueError('Value ({}) for variable {} out of range!'.format(v, self._v))
        return self._lpr[v]
    return 0.0

  def distribution(self):
    return 'multinomial'

  def expand(self):
    s = Sum()
    for i, p in enumerate(self._lpr):
      s.add(Indicator(self._v, i), p)
    return s

  def expand_str(self):
    s = '{},BINNODE,{}'.format(self._id, self._v)
    for i, l in enumerate(self._lpr):
      s += ',{}'.format(math.exp(self._lpr[i]))
    s += '\n'
    return s

