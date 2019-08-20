import numpy as np

from .node import Node, VarSet

class Product(Node):
  def __init__(self, id: int = None):
    Node.__init__(self, id)

  def eval(self, e: VarSet) -> float:
    return np.array([ c.eval(e) for c, i in enumerate(self._ch) ]).sum()

  def kind(self):
    return 'product'
