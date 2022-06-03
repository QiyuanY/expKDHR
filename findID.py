import numpy as np
from load import table2mat

hh_edge = np.load('./data/hh_graph.npy')
hh_edge_adj = hh_edge - 390
hh_edge_adj = table2mat(hh_edge_adj, 805)

ss_edge = np.load('./data/ss_graph.npy')
ss_edge_adj = np.array(ss_edge)
ss_edge_adj = table2mat(ss_edge_adj, 390)

h_index_1, h_index_2 = np.nonzero(hh_edge_adj)
s_index_1, s_index_2 = np.nonzero(ss_edge_adj)