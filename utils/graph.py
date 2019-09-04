import math

import numpy as np
import matplotlib.pyplot as plt
import graph_tool
import graph_tool.generation
import graph_tool.flow
import graph_tool.draw
import graph_tool.util
import networkx as nx
import networkx.algorithms
import networkx.algorithms.community

import spn
import utils

class VertexMap:
  # Assume scope is ordered, but if that is not the case, order must be set to True.
  def __init__(self, scope, order = False):
    if order:
      scope.sort()
    self._n = len(scope)
    self._vtx_to_var = [*scope]
    self._var_to_vtx = {}
    for i, v in enumerate(scope):
      self._var_to_vtx[v] = i

  # v: vertex
  # x: variable
  def map(self, v: graph_tool.Vertex, x: int):
    v = int(v)
    self._vtx_to_var[v] = x
    self._var_to_vtx[x] = v

  # f : Vertices -> Variables
  def f(self, v: graph_tool.Vertex):
    return self._vtx_to_var[v]

  # g : Variables -> Vertices
  def g(self, x: int):
    return self._var_to_vtx[x]

def _generate_sample_graph(D: spn.Dataset, S: spn.Scope, M: VertexMap, sample_func):
  G = graph_tool.Graph(directed = False)
  W = G.new_edge_property('double')
  G.edge_properties['weights'] = W
  G.add_vertex(n=len(S))
  for v in S:
    N = sample_func(S, v)
    for u in N:
      s = M.g(v)
      t = M.g(u)
      if G.edge(s, t) is None:
        e = G.add_edge(s, t)
        W[e] = __correlation(D, s, t)
  return G

def _generate_complete_graph(D: spn.Dataset, S: spn.Scope):
  n = len(S)
  G = graph_tool.generation.complete_graph(n, self_loops = False, directed = False)
  W = G.new_edge_property('double')
  G.edge_properties['weights'] = W
  for e in G.edges():
    s = int(e.source())
    t = int(e.target())
    if e not in W:
      W[e] = _correlation(D, s, t)
  return G

def _cut_mins(G: graph_tool.Graph) -> graph_tool.Graph:
  W = G.edge_properties['weights']

# D: dataset
# x: variable, st x <- x - min_x Index(Variables)
# y: variable, st y <- y - min_y Index(Variables)
# If scope is always in increasing order, then x and y are exactly the vertices from VertexMap
def _correlation(D: spn.Dataset, x: int, y: int) -> float:
  X = D[:,x]
  Y = D[:,y]

  x_mean = X.mean()
  y_mean = Y.mean()
  X_diff = X-x_mean
  Y_diff = Y-y_mean
  top = np.multiply(X_diff, Y_diff).sum()

  x_sq = math.sqrt(np.multiply(X_diff, X_diff).sum())
  y_sq = math.sqrt(np.multiply(Y_diff, Y_diff).sum())
  bottom = x_sq*y_sq

  if top == 0 and bottom == top:
    return 0.0
  return abs(top/bottom)

def _to_directed(G: graph_tool.Graph) -> graph_tool.Graph:
  H = graph_tool.Graph(directed = True)
  W = G.edge_properties['weights']
  U = H.new_edge_property('double')
  H.edge_properties['weights'] = U
  for e in G.edges():
    s, t = int(e.source()), int(e.target())
    e1 = H.add_edge(s, t, add_missing = True)
    e2 = H.add_edge(t, s)
    w = W[e]
    U[e1] = w
    U[e2] = w
  return H

def _to_hammock(G: graph_tool.Graph) -> [graph_tool.Graph, graph_tool.Vertex, graph_tool.Vertex]:
  H = _to_directed(G)
  W = H.edge_properties['weights']
  s = H.add_vertex()
  t = H.add_vertex()
  for v in H.vertices():
    if v != s and v != t:
      W[H.add_edge(s, v)] = 1.0
      W[H.add_edge(v, t)] = 1.0
  return H, s, t

class PartitionGraph:
  # Constructor for Graph.
  # Parameters:
  #  D:           dataset
  #  S:           scope
  #  sample_func: function : [VarID, Scope] -> List[VarID] of sampled neighbors given each variable
  #               if sample_func is None, then generate complete graph instead
  def __init__(self, D: spn.Dataset, S: spn.Scope, sample_func = None):
    self._G = None
    self._S = np.array(S)
    self._M = VertexMap(S)
    if sample_func is None:
      self._G = _generate_complete_graph(D, S)
    else:
      self._G = _generate_sample_graph(D, S, self._M, sample_func)

  def partition(self):
    W = self._G.edge_properties["weights"]
    _, P = graph_tool.flow.min_cut(self._G, W)
    label = self._G.new_edge_property('string')
    graph_tool.map_property_values(W, label, lambda w: str(w))
    graph_tool.draw.graph_draw(self._G, edge_pen_width=W, vertex_fill_color=P, edge_text=label,
                               output='/tmp/min_cut.pdf')
    P1, P2 = [], []
    for v, which in enumerate(P):
      x = self._M.f(v)
      if which:
        P1.append(x)
      else:
        P2.append(x)
    return P1, P2

  def partition_flow(self):
    H, s, t = _to_hammock(self._G)
    W = H.edge_properties["weights"]
    R = graph_tool.flow.push_relabel_max_flow(H, s, t, W)
    P = graph_tool.flow.min_st_cut(H, s, W, R)
    label = H.new_edge_property('string')
    graph_tool.map_property_values(W, label, lambda w: str(w))
    graph_tool.draw.graph_draw(H, edge_pen_width=W, vertex_fill_color=P, edge_text=label,
                               output='/tmp/min_cut.pdf')
    P1, P2 = [], []
    for v, which in enumerate(P):
      if v != s and v != t:
        x = self._M.f(v)
        if which:
          P1.append(x)
        else:
          P2.append(x)
    return P1, P2

  def partition_mst(self):
    W = self._G.edge_properties['weights']
    nW = self._G.new_edge_property('double')
    self._G.edge_properties['negative_weights'] = nW
    nW.a = list(-W.get_array())
    T = graph_tool.topology.min_spanning_tree(self._G, nW)
    H = graph_tool.Graph(directed = False)
    for i, v in enumerate(T):
      if v == 1:
        e = graph_tool.util.find_edge(self._G, self._G.edge_index, int(i))[0]
        H.add_edge(e.source(), e.target())
    I = np.nonzero(T.a)
    K = np.squeeze(np.dstack((I, np.array(W.a)[I])))
    # Sort by second column.
    E = K[K[:,1].argsort()]
    P = []
    for i, p in enumerate(E):
      e = graph_tool.util.find_edge(H, H.edge_index, int(i))[0]
      H.remove_edge(e)
      C, h = graph_tool.topology.label_components(H)
      P.append([self._S[p] for p in utils.group_by(np.array(C.a))])
    return P

def _generate_complete_network(D: spn.Dataset, S: spn.Scope):
  G = nx.complete_graph(S)
  for i, u in enumerate(S):
    for j, v in enumerate(S[i+1:]):
      G.add_edge(u, v, weight=_correlation(D, i, j+i+1))
  return G

class PartitionNetwork:
  # NetworkX variation for Kernighan-Lin Bisection
  def __init__(self, D: spn.Dataset, S: spn.Scope):
    self._S = S
    self._G = _generate_complete_network(D, S)

  def partition_balanced(self):
    return nx.algorithms.community.kernighan_lin_bisection(self._G, max_iter=1000000)

  def partition(self, draw: bool = False, n = None):
    if draw:
      pos = nx.spring_layout(self._G)
    P = []
    if n is None:
      for i in range(1, int(len(self._S)/2)+1):
        pi = nx.algorithms.community.kernighan_lin_bisection(self._G, partition=(self._S[:i],
                                                                                 self._S[i:]),
                                                             max_iter=100)
        P.append((list(pi[0]), list(pi[1])))
        if draw:
          nx.draw_networkx_nodes(self._G, pos=pos, nodelist=P[0], node_color='blue')
          nx.draw_networkx_nodes(self._G, pos=pos, nodelist=P[1], node_color='green')
          nx.draw_networkx_edge_labels(self._G, pos=pos, edge_labels=nx.get_edge_attributes(self._G,
                                                                                            'weight'),
                                       font_size=5)
          nx.draw_networkx_edges(self._G, pos=pos)
          plt.show()
    else:
      n = int(len(self._S)/n)
      pi = nx.algorithms.community.kernighan_lin_bisection(self._G, partition=(self._S[:n],
                                                                               self._S[n:]),
                                                           max_iter = 100)
      P.append((list(pi[0]), list(pi[1])))
    return P
