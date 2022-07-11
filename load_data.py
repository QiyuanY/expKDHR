from utils import *
from torch_geometric.data import Data
from load import table2mat
from sklearn.model_selection import train_test_split
import math

seed = 2021512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


class DataLoad(object):
    def __init__(self, para):
        self.prescript = pd.read_csv('./data/prescript_1195.csv', encoding='utf-8')
        self.pLen = len(self.prescript)  # 数据集的数量
        self.para = para
        self.data = []

    def GetIndex(self):
        """创建3种图数据"""
        # 读取S-H图
        sh_edge = np.load('./data/sh_graph.npy')
        sh_edge = sh_edge.tolist()
        sh_edge_index = torch.tensor(sh_edge, dtype=torch.long)
        sh_x = torch.tensor([[i] for i in range(1195)], dtype=torch.float)
        sh_data = self.Indexrank(Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).edge_index)  ### 制图
        sh_data_origin = Data(x=sh_x, edge_index=sh_edge_index.t().contiguous()).edge_index

        # S-S G
        ss_edge = np.load('./data/ss_graph.npy')
        ss_edge_adj = np.array(ss_edge)
        ss_edge_adj = table2mat(ss_edge_adj, 390)

        # H-H G
        hh_edge = np.load('./data/hh_graph.npy')
        hh_edge_adj = hh_edge - 390
        hh_edge_adj = table2mat(hh_edge_adj, 805)

        return ss_edge_adj, hh_edge_adj, sh_data, sh_data_origin

    def GetSet(self):
        train, dev_test = train_test_split(self.data, test_size=(self.para.dev_ratio + self.para.test_ratio),
                                           shuffle=False,
                                           random_state=2021)
        dev, test = train_test_split(dev_test, test_size=1 - 0.5, shuffle=False, random_state=2021)
        return train, dev, test

    def GetDataset(self):
        # 症状的one-hot 矩阵
        pS_list = [[0] * 390 for _ in range(self.pLen)]
        pS_array = np.array(pS_list)
        # 草药的one-hot 矩阵
        pH_list = [[0] * 805 for _ in range(self.pLen)]
        pH_array = np.array(pH_list)

        pS_array = torch.from_numpy(pS_array).to(device).float()
        pH_array = torch.from_numpy(pH_array).to(device).float()

        # 迭代数据集， 赋值  ###目前看不懂这里
        for i in range(self.pLen):
            j = eval(self.prescript.iloc[i, 0])
            pS_array[i, j] = 1

            k = eval(self.prescript.iloc[i, 1])
            k = [x - 390 for x in k]
            pH_array[i, k] = 1

        # 训练集开发集测试集的下标
        p_list = [x for x in range(self.pLen)]
        self.data = p_list
        x_train, x_dev, x_test = self.GetSet()

        train_dataset = presDataset(pS_array[x_train], pH_array[x_train])
        dev_dataset = presDataset(pS_array[x_dev], pH_array[x_dev])
        test_dataset = presDataset(pS_array[x_test], pH_array[x_test])

        return train_dataset, dev_dataset, test_dataset

    def Indexrank(self, data):
        data = data.numpy()
        tmp = data[0, :]
        for i in range(79870):
            if tmp[i] < 390:
                data[0][i], data[1][i] = data[1][i], data[0][i]
        return data
