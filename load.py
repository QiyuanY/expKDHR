import numpy as np
import torch
from torch_geometric.data import Data


#
# input_data = np.load(r"./data/ss_graph.npy")
# data = input_data.tolist()

def load_data(path, dim):
    sh_edge = np.load(path)
    sh_edge = sh_edge.tolist()
    sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
    sh_x = torch.tensor([[i] for i in range(dim, )], dtype=torch.float)
    sh_data = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous())  ### 制图
    # sh_data_adj = SparseTensor(row=sh_data.edge_index[0], col=sh_data.edge_index[1],
    #                            sparse_sizes=(1195, 1195))  ### 邻接矩阵
    return sh_data
