import math

from .node import Node

def save(S: Node, filename: str):
  with open(filename, 'w') as f:
    f.write('##NODES##\n')
    Q = S.top_down_order()
    for n in Q:
      if n.kind() == 'sum':
        f.write('{},SUM\n'.format(n._id))
      elif n.kind() == 'product':
        f.write('{},PRD\n'.format(n._id))
      else: # Leaf node
        if n.distribution() == 'multinomial':
          f.write(n.expand_str())
    f.write('##EDGES##\n')
    for n in Q:
      if n.kind() == 'sum':
        for i, c in enumerate(n._ch):
          f.write('{},{},{}\n'.format(n._id, c._id, math.exp(n._lw[i])))
      elif n.kind() == 'product':
        for i, c in enumerate(n._ch):
          f.write('{},{}\n'.format(n._id, c._id))

