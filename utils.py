#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong
import pickle
import parameter
import numpy as np
import pandas as pd
import random
import torch
from torch_geometric.data import Data
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import time

seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

def save_obj(obj, name ):
    with open(name + '.pkl', 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_obj(name ):
    with open( name + '.pkl', 'rb') as f:
        return pickle.load(f)

class presDataset(torch.utils.data.Dataset):
    def __init__(self, a, b):
        self.pS_array, self.pH_array = a, b
    def __getitem__(self, idx):
        sid = self.pS_array[idx]
        hid = self.pH_array[idx]
        return sid, hid

    def __len__(self):
        return self.pH_array.shape[0]


def Drawpic(epoch, data1, data2, data3, para):

    x = range(epoch)
    plt.plot(x, data1, color='blue', label='precision')
    plt.plot(x, data2, color='green', label='recall')
    plt.plot(x, data3, color='red', label = 'f1score')

    plt.legend()
    plt.title(para)
    plt.xlabel('epoch')
    plt.ylabel('rate')
    plt.savefig('res_{}.png'.format(time.strftime("%Y%m%d_%H%M%S")))


