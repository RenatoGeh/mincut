from .node import Node

class Leaf(Node):
  def __init__(self, id: int = None):
    Node.__init__(self, id)

  def distribution(self):
    raise NotImplementedError('Leaf has no distribution!')

  def kind(self):
    return 'leaf'
