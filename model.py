#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong
import faiss
import numpy as np
import pandas as pd
import random
import torch
import torch.nn.functional as F
from torch_geometric.nn import MessagePassing
from sklearn.cluster import KMeans
from reckit import randint_choice
import scipy.sparse as sp
from load import mat2table

seed = 2021
np.random.seed(seed)
if torch.cuda.is_available():
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
torch.manual_seed(seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class GCNConv_SH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SH, self).__init__(aggr='mean')  # 对邻居节点特征进行平均操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # x has shape [N, in_channels]
        # edge_index has shape [2, E]
        # 公式2
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)  # m = e*T 公式1
        return x_j


"""
替换对比损失
"""


class GCNConv_SS_HH(MessagePassing):
    def __init__(self, in_channels, out_channels):
        super(GCNConv_SS_HH, self).__init__(aggr='add')  # 对邻居节点特征进行sum操作
        self.lin = torch.nn.Linear(in_channels, out_channels)
        self.tanh = torch.nn.Tanh()

    def forward(self, x, edge_index):
        # 公式10
        out = self.propagate(edge_index, x=x)
        return self.tanh(out)

    def message(self, x_j):
        x_j = self.lin(x_j)
        return x_j


def sp_mat_to_sp_tensor(sp_mat):
    coo = sp_mat.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.asarray([coo.row, coo.col]))
    return torch.sparse_coo_tensor(indices, coo.data, coo.shape).coalesce()


class KDHR(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, edge_matrix_1, edge_matrix_2, edge_index, batchSize,
                 drop):
        super(KDHR, self).__init__()
        # if trail:
        #     pass

        self.batchSize = batchSize
        self.dropout = drop
        self.SH_embedding = torch.nn.Embedding(sh_num, embedding_dim)
        # s和h的初始表示 0层
        self.S_embedding = torch.nn.Embedding(ss_num, embedding_dim)
        self.H_embedding = torch.nn.Embedding(hh_num, embedding_dim)

        self.p_embedding_0 = []
        self.p_embedding_1 = []

        self.c_embedding_0 = []
        self.c_embedding_1 = []

        self.g1_emb0 = []
        self.g1_emb1 = []

        self.g2_emb0 = []
        self.g2_emb1 = []

        self.ssl_temp = 0.5
        self.ssl_reg = 1e-6
        self.ssl_ratio = 0.5
        self.mask_1 = 200
        self.mask_2 = 150
        self.alpha = 1.5
        self.latent_dim = 64
        self.k = 20
        self.k_1 = 9
        self.proto_reg = 8e-8
        self.con_reg = 0.1
        self.device = device
        self.g = []

        self.num_users = 390
        self.num_items = 805
        self.ssl_aug_type = 'ed'

        self.ssl_u, self.ssl_i = [], []
        self.nce_u, self.nce_i = [], []
        self.con_u, self.con_i = [], []
        self.sgl_u, self.sgl_i = [], []

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.userID = torch.linspace(0, 804, 805).long()
        self.itemID = torch.linspace(0, 389, 390).long()
        self.xID = torch.linspace(0, 1194, 1195).long()

        self.edge_u = edge_matrix_1
        self.edge_i = edge_matrix_2

        self.edge_index = edge_index

        # S-H 图所需的网络
        # S
        self.convSH_TostudyS_1 = GCNConv_SH(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2 = GCNConv_SH(embedding_dim, embedding_dim)
        self.SH_mlp_1 = torch.nn.Linear(embedding_dim, 64)
        self.SH_bn_1 = torch.nn.BatchNorm1d(64)
        self.SH_tanh_1 = torch.nn.Tanh()
        # H
        self.convSH_TostudyS_1_h = GCNConv_SH(embedding_dim, embedding_dim)
        self.convSH_TostudyS_2_h = GCNConv_SH(embedding_dim, embedding_dim)
        self.SH_mlp_1_h = torch.nn.Linear(embedding_dim, 64)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(64)
        self.SH_tanh_1_h = torch.nn.Tanh()

        # SUM
        self.mlp = torch.nn.Linear(embedding_dim, 64)
        # cat
        # self.mlp = torch.nn.Linear(512, 512)
        self.SI_bn = torch.nn.BatchNorm1d(64)
        self.relu = torch.nn.ReLU()

        self.proj = torch.nn.Sequential(
            torch.nn.Linear(embedding_dim, 64),
            torch.nn.ELU(),
            torch.nn.Linear(embedding_dim, 64)
        )
        self.tau = 0.5
        self.lam = 0.5
        for model in self.proj:
            if isinstance(model, torch.nn.Linear):
                torch.nn.init.xavier_normal_(model.weight, gain=1.414)

        # edge_drop
        self.total_g = self.edge_index
        self.sub_g1, self.sub_g2 = torch.as_tensor(self.SGL(), device=device)

    def forward(self, edge_index_SH, prescription):
        # S-H图搭建
        l = torch.tensor(np.arange(0, 1195), dtype=torch.float32, requires_grad=True).to(device).long()

        x_SH1 = self.SH_embedding(l).to(device)
        # 第一层
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), self.total_g)
        # 第二层
        x_SH6 = self.convSH_TostudyS_2(x_SH2, self.total_g)

        SH_g1 = self.SH_embedding(l).to(device)
        # 第一层
        SH_g2 = self.convSH_TostudyS_1(SH_g1.float(), self.sub_g1)
        # 第二层
        SH_g3 = self.convSH_TostudyS_2(SH_g2, self.sub_g1)

        SH_g11 = self.SH_embedding(l).to(device)
        # 第一层
        SH_g22 = self.convSH_TostudyS_1(SH_g11.float(), self.sub_g2)
        # 第二层
        SH_g33 = self.convSH_TostudyS_2(SH_g22, self.sub_g2)

        self.p_embedding_0, self.p_embedding_1 = torch.split(x_SH1, [805, 390])
        self.c_embedding_0, self.c_embedding_1 = torch.split(x_SH6, [805, 390])

        self.g1_emb0, self.g1_emb1 = torch.split(SH_g3, [805, 390])

        self.g2_emb0, self.g2_emb1 = torch.split(SH_g33, [805, 390])

        # sum操作
        _, s_i, s_u = self.ssl_layer_loss()
        # _, n_u, n_i = self.ProtoNCE_loss()
        # _, n_i, n_u = self.high_loss()
        _, n_i, n_u = self.sgl_loss()
        self.ssl_u = s_u
        self.nce_u = n_u
        self.ssl_i = s_i
        self.nce_i = n_i
        _, c_u, c_i = self.Same2Loss()

        # n_i:805 n_u:390
        # es = torch.as_tensor(s_u + n_u, device=device)
        # eh = torch.as_tensor(s_i + n_i, device=device)

        # SI 集成多个症状为一个症状表示 batch*390 390*dim => batch*dim
        # es = es.view(390, -1)
        es = c_u.view(390, -1)
        e_synd = torch.mm(prescription, es)  # prescription * es
        # batch*1
        preSum = prescription.sum(dim=1).view(-1, 1)
        # batch*dim
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        # e_synd_norm = e_synd_norm.view(-1, 256)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # batch*dim
        # batch*dim dim*805 => batch*805
        eh = c_i.view(805, -1)
        # eh = eh.view(805, -1)

        pre = torch.mm(e_synd_norm, eh.t())
        return pre

    def con_loss(self, emb1, emb2, emb_all, model):
        """

        :param emb1:
        :param emb2:
        :param emb_all: 其他所有embeddings
        :param model:选择进行的对比学习方式（公式不同）
        :return:对比学习的loss
        """
        res = 0
        pos_score = torch.mul(emb1, emb2).sum(dim=1)
        ttl_score = torch.matmul(emb1, emb_all.transpose(0, 1))

        if model == 'high':
            # 分子分母分别累加
            pos_score = torch.exp(pos_score / self.ssl_temp).sum()
            ttl_score = torch.exp(ttl_score / self.ssl_temp).sum()  # 这里有个地方没搞懂

            res = -torch.log(pos_score / ttl_score)
        elif model == 'ssl':
            # 分母累加后结果累加
            pos_score = torch.exp(pos_score / self.ssl_temp)
            ttl_score = torch.exp(ttl_score / self.ssl_temp).sum(dim=1)

            res = -torch.log(pos_score / ttl_score).sum()

        return res

    def ssl_layer_loss(self):  # KDHR的0层和N层
        """
        其中的含dim的Norm函数所执行的操作是将emb格式转换为tensor格式并进行正则，如果没有给定dim，则只进行正则，其中n_dim表示对不同的维度进行正则
        :return:
        """
        model = 'ssl'
        current_user_embeddings, current_item_embeddings = self.c_embedding_0, self.c_embedding_1
        previous_user_embeddings_all, previous_item_embeddings_all = self.p_embedding_0, self.p_embedding_1

        current_user_embeddings = self.Norm(current_user_embeddings[self.userID], 805, n_dim=0)
        current_item_embeddings = self.Norm(current_item_embeddings[self.itemID], 390, n_dim=1)
        previous_user_embeddings_all = self.Norm(previous_user_embeddings_all[self.userID], 805, n_dim=0)
        previous_item_embeddings_all = self.Norm(previous_item_embeddings_all[self.itemID], 390, n_dim=1)

        norm_user_emb1, norm_user_emb2 = current_user_embeddings, previous_user_embeddings_all
        norm_item_emb1, norm_item_emb2 = current_item_embeddings, previous_item_embeddings_all

        ssl_loss_user = self.con_loss(norm_user_emb1, norm_user_emb2, previous_user_embeddings_all, model=model)

        ssl_loss_item = self.con_loss(norm_item_emb1, norm_item_emb2, previous_item_embeddings_all, model=model)

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss, norm_user_emb2, norm_item_emb2

    def sgl_loss(self):
        """

        :return:ssl_loss, user, item
        """
        model = 'ssl'

        g1_user_emb, g2_user_emb = self.g1_emb0, self.g2_emb0
        g1_item_emb, g2_item_emb = self.g1_emb1, self.g2_emb1

        g1_user_emb = self.Norm(g1_user_emb[self.userID], 805, n_dim=0)
        g2_user_emb = self.Norm(g2_user_emb[self.userID], 805, n_dim=0)
        g1_item_emb = self.Norm(g1_item_emb[self.itemID], 390, n_dim=1)
        g2_item_emb = self.Norm(g2_item_emb[self.itemID], 390, n_dim=1)

        norm_user_emb1, norm_user_emb2 = g1_user_emb, g2_user_emb
        norm_item_emb1, norm_item_emb2 = g1_item_emb, g2_item_emb

        sgl_loss_user = self.con_loss(norm_user_emb1, norm_user_emb2, g1_user_emb, model=model)

        sgl_loss_item = self.con_loss(norm_item_emb1, norm_item_emb2, g1_item_emb, model=model)
        sgl_loss = self.ssl_reg * (sgl_loss_user + self.alpha * sgl_loss_item)

        return sgl_loss, g1_user_emb, g1_item_emb

    def run_kmeans(self, x, dim, _k):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        # Kmeans这个方法的输出是什么含义？
        kmeans = faiss.Kmeans(d=self.latent_dim, k=_k, gpu=True)
        x = np.reshape(x, [dim, self.latent_dim])
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, i = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.tensor(cluster_cents).to(device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(i).squeeze().to(self.device)
        return centroids, node2cluster

    def e_step(self):
        user_embeddings = self.c_embedding_0.detach().cpu().numpy()
        item_embeddings = self.c_embedding_1.detach().cpu().numpy()
        # user_embeddings = self.c_embedding_0.weight.detach().cpu().numpy()
        # item_embeddings = self.c_embedding_1.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, dim=805, _k=self.k)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings, dim=390, _k=self.k_1)

    def ProtoNCE_loss(self):
        """

        :return: kmeans_loss, item, user
        """
        # user_embeddings_all, item_embeddings_all = torch.split(node_embedding, [self.n_users, self.n_items])

        # self.c_embedding = torch.tensor(self.c_embedding)
        # self.p_embedding = torch.tensor(self.p_embedding)
        user_embeddings_all, item_embeddings_all = self.c_embedding_0, self.c_embedding_1
        # user_embeddings = user_embeddings_all[user]  # [B, e]
        user_embeddings_all = torch.as_tensor(user_embeddings_all)
        item_embeddings_all = torch.as_tensor(item_embeddings_all)

        norm_user_embeddings = F.normalize(user_embeddings_all, dim=0)
        norm_user_embeddings = norm_user_embeddings.view(805, -1)
        self.e_step()
        user2cluster = self.user_2cluster[self.userID]  # [B,]

        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.matmul(norm_user_embeddings, self.user_centroids.transpose(0, 1))
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        proto_nce_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        # item_embeddings = item_embeddings_all[item]
        norm_item_embeddings = F.normalize(item_embeddings_all, dim=0)
        norm_item_embeddings = norm_item_embeddings.view(390, -1)
        item2cluster = self.item_2cluster[self.itemID]  # [B, ]
        item2centroids = self.item_centroids[item2cluster]  # [B, e]
        pos_score_item = torch.mul(norm_item_embeddings, item2centroids).sum(dim=1)
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.matmul(norm_item_embeddings, self.item_centroids.transpose(0, 1))
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)
        proto_nce_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        proto_nce_loss = self.proto_reg * (proto_nce_loss_user + proto_nce_loss_item)
        return proto_nce_loss, norm_item_embeddings, norm_user_embeddings

    def sim(self, z1, z2):
        z1_norm = torch.norm(z1, dim=-1, keepdim=True)
        z2_norm = torch.norm(z2, dim=-1, keepdim=True)
        dot_numerator = torch.mm(z1, z2.t())
        dot_denominator = torch.mm(z1_norm, z2_norm.t())
        sim_matrix = torch.exp(dot_numerator / dot_denominator / self.tau)
        return sim_matrix

    def SameContrast(self, emb1, emb2, pos):
        z_proj_mp = self.proj(emb1)
        z_proj_sc = self.proj(emb2)
        matrix_mp2sc = self.sim(z_proj_mp, z_proj_sc)
        matrix_sc2mp = matrix_mp2sc.t()
        matrix_mp2sc = matrix_mp2sc / (torch.sum(matrix_mp2sc, dim=1).view(-1, 1) + 1e-8)
        # pos 不可用，学习上边的分母构造方法
        lori_mp = -torch.log(matrix_mp2sc.mul(pos + 1e-13).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos + 1e-13).sum(dim=-1)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc, emb1, emb2

    def findID(self):
        """

        :return: user[1:2] item[1:2]
        """
        h_index = np.nonzero(self.edge_i)
        s_index = np.nonzero(self.edge_u)

        return s_index[0], s_index[1], h_index[0], h_index[1]

    def Same2Loss(self):
        same_loss_user, _, __ = self.SameContrast(self.ssl_u, self.nce_u, self.edge_u)
        same_loss_item, _, __ = self.SameContrast(self.ssl_i, self.nce_i, self.edge_i)
        user = torch.as_tensor(self.ssl_u + self.nce_u)
        item = torch.as_tensor(self.ssl_i + self.nce_i)
        return self.con_reg * torch.as_tensor(same_loss_user + same_loss_item), user, item

    def calculate_loss(self):
        """

        :return: ssl + high
        """
        ssl_loss, self.ssl_i, self.ssl_u = self.ssl_layer_loss()
        print(ssl_loss)
        sgl_loss, self.sgl_u, self.sgl_i = self.sgl_loss()
        print(sgl_loss)
        # high_loss, self.nce_u, self.nce_i = self.high_loss()
        # print(high_loss)
        # nce_loss, self.nce_u, self.nce_i = self.ProtoNCE_loss()
        # print(nce_loss)
        # contrast_loss = self.con_reg * torch.as_tensor(self.con_i + self.con_u, device=device).float()
        # con_loss, self.con_u, self.con_i = self.Same2Loss()
        # print(con_loss)

        # return nce_loss + con_loss

        return ssl_loss + sgl_loss

    def Norm(self, emb, dim=0, n_dim=0):
        """
        :param emb: 需要正则的emb
        :param dim: 该emb维度
        :param n_dim: 需要对第n维度进行正则(这里user为0，item为1)
        :return: 正则后的emb
        """
        res_emb = emb
        if not torch.is_tensor(res_emb):
            res_emb = torch.as_tensor(res_emb, device=device).view(dim, -1)
        res_emb = F.normalize(res_emb, dim=n_dim)

        return res_emb

    def high_loss(self):
        """
        :return:loss, user, item
        """
        model = 'high'
        index_1, index_2, index_3, index_4 = self.findID()
        user_norm = self.Norm(self.c_embedding_0, dim=805, n_dim=0)
        item_norm = self.Norm(self.c_embedding_1, dim=390, n_dim=1)
        user_emb1, user_emb2 = user_norm[index_1], user_norm[index_2]
        item_emb1, item_emb2 = item_norm[index_3], item_norm[index_4]

        ssl_loss_user = self.con_loss(user_emb1, user_emb2, user_norm, model=model)
        ssl_loss_item = self.con_loss(item_emb1, item_emb2, item_norm, model=model)

        high_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)

        return high_loss, user_norm, item_norm

    def SGL(self):
        """

        :return:SGL所得的两个子图（经过对高频节点的边进行了drop处理）
        """
        sub_graph1 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)
        sub_graph2 = self.create_adj_mat(is_subgraph=True, aug_type=self.ssl_aug_type)

        return sub_graph1, sub_graph2

    def create_adj_mat(self, is_subgraph=False, aug_type='ed'):
        n_nodes = self.num_users + self.num_items
        users_items = self.total_g.cpu()
        res = np.argsort(users_items[1, :])
        # 2*n 0:0-390 1
        tmp = users_items[:, res].numpy()
        cnt1 = pd.Series(tmp[0, :])
        # 对出现频次进行排序
        srt = cnt1.value_counts()
        # 将行索引变为一列值
        srt = srt.reset_index()
        # 转回numpy格式
        t1 = np.array(srt)
        # 取出频率高于200的索引
        t1 = t1[np.where(t1[:, 1] > 200)][:, 0]
        temp = []
        ttemp = []
        for i in range(79870):
            # tmp.shape[1]
            if tmp[0][i] in t1:
                temp.append((tmp[0][i], tmp[1][i]))
        # print(temp)
        temp = np.array(temp)

        cnt2 = pd.Series(temp[:][:, 1])
        srt = cnt2.value_counts()
        srt = srt.reset_index()
        # srt = srt.reset_index()
        t2 = np.array(srt)
        t2 = t2[np.where(t2[:, 1] > 100)][:, 0]
        for i in range(len(temp)):
            if temp[i][1] in t2:
                ttemp.append((temp[i][0], temp[i][1]))
        # 总数据的转置
        tmp = tmp.transpose()
        total = []

        # 将数据转为list进行切割重组
        for i in range(79870):
            total.append((tmp[i][0], tmp[i][1]))
        # total = pd.DataFrame(total).to_numpy()
        for i in range(len(ttemp)):
            if ttemp[i] in total:
                total.remove(ttemp[i])
        length = len(ttemp)
        ttemp = ttemp + total
        result = np.array(ttemp)
        result = result.reshape([2,79870])

        users_np, items_np = result[0, :], result[1, :]
        # users_np, items_np = users_items[0, :], users_items[1, :]
        adj_matrix = []
        if is_subgraph and self.ssl_ratio > 0:
            if aug_type in ['ed', 'rw']:
                # keep_idx = randint_choice(len(users_np), size=int(len(users_np) * (1 - self.ssl_ratio)), replace=False)
                keep_idx = randint_choice(length, size=int(length * (1 - self.ssl_ratio)), replace=False)
                user_np = np.array(users_np)[keep_idx]
                item_np = np.array(items_np)[keep_idx]
                adj_matrix = np.vstack((user_np, item_np))
        else:
            adj_matrix = np.vstack((users_np, items_np))


        return adj_matrix
