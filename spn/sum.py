from scipy.special import logsumexp

from .node import Node, VarSet

class Sum(Node):
  def __init__(self, id: int = None):
    Node.__init__(self, id)
    self._lw = []

  # Adds a new weighted child. Expects weight in logspace.
  def add(self, c: Node, lw: float):
    Node.add(self, c)
    self._lw.append(lw)

  def eval(self, e: VarSet) -> float:
    return logsumexp([ c.eval(e) + self._lw[i] for c, i in enumerate(self._ch) ])

  def kind(self):
    return 'sum'
