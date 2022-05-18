import numpy as np
import scipy.sparse as sp
# import torch
# import torch_sparse
# from torch_geometric.data import Data
import scipy.sparse as sp


# def arr2coo(arr):
#     a = np.array(arr)
#     idx = a.nonzero()  # (row, col)
#     data = a[idx]
#
#     # to torch tensor
#     idx_t = torch.LongTensor(np.vstack(idx))
#     data_t = torch.FloatTensor(data)
#     coo_a = torch.sparse_coo_tensor(idx_t, data_t, a.shape)
#     return coo_a


def table2matrix(table):
    ret = []
    N = len(table)
    for i in range(N):
        tmp = [0] * N
        for j in table[i]:
            tmp[j] = 1
        ret.append(tmp)
    return ret


def pos_create_1(path, p):
    input_data = np.load(path)
    data = input_data
    mat = np.array(table2matrix(data))
    np.save("ss_pos.npy", mat)


def pos_create_2(path, p):
    input_data = np.load(path)
    data = input_data - 390
    mat = np.array(table2matrix(data))
    np.save("hh_pos.npy", mat)


def table2mat(arr, dim):
    ret = np.zeros(shape=(dim, dim))
    for i in range(arr.shape[0]):
        ret[arr[i][0],arr[i][1]] = 1
    return ret


# pos_create_1('./data/ss_graph.npy', 390)
# pos_create_2('./data/hh_graph.npy', 805)

# ss_edge = np.load('./data/hh_graph.npy') - 390
# # a = ss_edge[:,[0]] - 390
# # b = ss_edge[:,[1]]
# # ss_edge = np.concatenate((a,b),axis=1)
# # print(ss_edge[0][0])
# # print(ss_edge[0][1])
# #
# # print(ss_edge.shape[0])
# mat = table2mat(ss_edge, 805)
# print(mat)

# ss_edge_index = torch.tensor(ss_edge, dtype=torch.long)
# ss_x = torch.tensor([[i] for i in range(390)], dtype=torch.float)
# ss_data = Data(x=ss_x, edge_index=ss_edge_index.t().contiguous())

# hh_edge = np.load('./data/hh_graph.npy').tolist()
# hh_edge_index = torch.tensor(hh_edge, dtype=torch.long) - 390  # 边索引需要减去390
# hh_x = torch.tensor([[i] for i in range(390, 1195)], dtype=torch.float)
# hh_data = Data(x=hh_x, edge_index=hh_edge_index.t().contiguous())

# x_ss = ss_data.x.shape[0]
# edge_shape = np.zeros((x_ss, x_ss)).shape
# values = torch.tensor(np.ones(ss_data.edge_index.shape[1]), dtype=torch.long)
# ss_edge = torch.sparse_coo_tensor(ss_data.edge_index, values, edge_shape)

# def adj2coo(ss_ed):
#     tmp_coo = sp.coo_matrix(ss_ed)
#     values = tmp_coo.data
#     indices = np.vstack((tmp_coo.row, tmp_coo.col))
#     i = torch.tensor(indices, dtype=torch.long)  # -390
#     v = torch.tensor(values, dtype=torch.long)
#     edge_index = torch.sparse_coo_tensor(i, v, tmp_coo.shape)
#     np.save("ss_coo.npy", edge_index)


# adj2coo()
