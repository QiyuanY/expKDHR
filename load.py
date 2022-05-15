import numpy as np
import scipy.sparse as sp


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
