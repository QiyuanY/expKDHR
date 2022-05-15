import numpy as np
import scipy.sparse as sp
import torch

def arr2coo(arr):
    a = np.array(arr)
    idx = a.nonzero()  # (row, col)
    data = a[idx]

    # to torch tensor
    idx_t = torch.LongTensor(np.vstack(idx))
    data_t = torch.FloatTensor(data)
    coo_a = torch.sparse_coo_tensor(idx_t, data_t, a.shape)
    return coo_a

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

pos_create_1('./data/ss_graph.npy', 390)
pos_create_2('./data/hh_graph.npy', 805)
