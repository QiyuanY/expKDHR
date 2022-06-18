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
        ret[arr[i][0], arr[i][1]] = 1
    return ret


def mat2table(mat, dim):
    arr = [[], []]
    for i in range(dim):
        for j in range(dim):
            if mat[i][j] != 0:
                arr[0].append(i)
                arr[1].append(j)
    return arr

