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


class KDHR(torch.nn.Module):
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, edge_matrix_1, edge_matrix_2, batchSize, drop):
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

        self.ssl_temp = 0.5
        self.ssl_reg = 1e-6
        self.alpha = 1.5
        self.latent_dim = 64
        self.k = 20
        self.k_1 = 9
        self.proto_reg = 8e-8
        self.con_reg = 0.1
        self.device = device

        self.ssl_u, self.ssl_i = [], []
        self.nce_u, self.nce_i = [], []
        self.con_u, self.con_i = [], []

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.userID = torch.linspace(0, 804, 805).long()
        self.itemID = torch.linspace(0, 389, 390).long()
        self.xID = torch.linspace(0, 1194, 1195).long()

        self.edge_u = edge_matrix_1
        self.edge_i = edge_matrix_2

        # S-H 图所需的网络
        # S
        self.convSH_TostudyS_1 = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2 = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3 = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1 = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1 = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1 = torch.nn.Tanh()
        # H
        self.convSH_TostudyS_1_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.convSH_TostudyS_2_h = GCNConv_SH(embedding_dim, embedding_dim)

        # self.convSH_TostudyS_3_h = GCNConv_SH(embedding_dim, embedding_dim)

        self.SH_mlp_1_h = torch.nn.Linear(embedding_dim, 256)
        self.SH_bn_1_h = torch.nn.BatchNorm1d(256)
        self.SH_tanh_1_h = torch.nn.Tanh()
        # # S-S图网络
        # self.convSS = GCNConv_SS_HH(embedding_dim, 256)
        # # H-H图网络  维度加上嵌入KG特征的维度
        # self.convHH = GCNConv_SS_HH(embedding_dim+27, 256)
        # # self.convHH = GCNConv_SS_HH(embedding_dim, 256)
        # SI诱导层
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

    def forward(self, edge_index_SH, edge_index_SS, edge_index_HH, prescription):
        # S-H图搭建
        # 第一层
        l = torch.tensor(np.arange(0, 1195), dtype=torch.float32, requires_grad=True).to(device).long()
        x_SH1 = self.SH_embedding(l).to(device)
        # x_SH1 = torch.tensor(self.SH_embedding, dtype=torch.float)[self.xID]

        # 这里的x_SH2和下边的x_SH22有区别，目前不知道什么原因
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), edge_index_SH)
        # 第二层
        x_SH6 = self.convSH_TostudyS_2(x_SH2, edge_index_SH)
        # # x_SH1 = self.x_SH1.requires_grad_(True)
        # # x_SH6 = x_SH6.requires_grad_(True)
        # # x_SH6 = x_SH6.view(-1, 256)
        # # 第三层
        # # x_SH7 = self.convSH_TostudyS_3(x_SH6, edge_index_SH)
        #
        # # x_SH9 = (x_SH1 + x_SH2 + x_SH6 ) / 3.0
        # # x_SH9 = self.SH_mlp_1(x_SH9)
        # # x_SH9 = x_SH9.view(1195, -1)
        # # x_SH9 = self.SH_bn_1(x_SH9)
        # # x_SH9 = self.SH_tanh_1(x_SH9)
        #
        # # SH H
        # # 0: 草药 1: 症状
        #
        x_SH11 = self.SH_embedding(l.long()).to(device)
        x_SH22 = self.convSH_TostudyS_1_h(x_SH11.float(), edge_index_SH)
        # 第二层
        x_SH66 = self.convSH_TostudyS_2_h(x_SH22, edge_index_SH)
        # x_SH11 = self.x_SH11.requires_grad_(True)
        # x_SH66 = x_SH66.requires_grad_(True)
        # x_SH66 = x_SH66.view(-1, 256)
        self.p_embedding_0, self.p_embedding_1 = torch.split(x_SH11, [805, 390])
        self.c_embedding_0, self.c_embedding_1 = torch.split(x_SH6, [805, 390])

        _, s_i, s_u = self.ssl_layer_loss()
        # s_i:805, s_u:390
        # 第三层
        # x_SH77 = self.convSH_TostudyS_3_h(x_SH66, edge_index_SH)

        # x_SH99 = (x_SH11 + x_SH22 +x_SH66 ) / 3.0
        # x_SH99 = self.SH_mlp_1_h(x_SH99)
        # x_SH99 = x_SH99.view(1195, -1)
        # x_SH99 = self.SH_bn_1_h(x_SH99)
        # x_SH99 = self.SH_tanh_1_h(x_SH99)

        # # S-S图搭建
        # x_ss0 = self.SH_embedding(x_SS.long())
        # # = self.convSS(x_ss0.float(), edge_index_SS) # S_S图中 s的嵌入
        # x_loss_ss, s_lossemb = self.same2ssl(x_ss0, edge_index_SS)
        # x_ss1 = x_ss0.view(390, -1)
        # # H-H图搭建
        # x_hh0 = self.SH_embedding(x_HH.long())
        # x_hh0 = x_hh0.view(-1, 64)
        # # x_hh0 = torch.cat((x_hh0.float(), kgOneHot), dim=-1)
        # x_loss_hh, h_lossemb = self.same2ssl(x_hh0, edge_index_HH)
        # # x_hh1 = self.convHH(x_hh0.float(), edge_index_HH)  # H_H图中 h的嵌入
        # x_hh1 = x_hh0.view(805, -1)

        # 信息融合

        # sum操作
        _, n_u, n_i = self.ProtoNCE_loss()
        self.ssl_u = s_u
        self.nce_u = n_u
        self.ssl_i = s_i
        self.nce_i = n_i
        _, c_u, c_i = self.Same2Loss()

        # self.con_u = self.SameContrast(s_u, n_u, edge_index_SS)
        # self.con_i = self.SameContrast(s_i, n_i, edge_index_HH)
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

    def ssl_layer_loss(self):  # KDHR的0层和N层
        # self.c_embedding_0 = torch.tensor(self.c_embedding_0)
        # self.p_embedding_0 = torch.tensor(self.p_embedding_0)

        current_user_embeddings, current_item_embeddings = self.c_embedding_0, self.c_embedding_1
        previous_user_embeddings_all, previous_item_embeddings_all = self.p_embedding_0, self.p_embedding_1

        current_item_embeddings = torch.as_tensor(current_item_embeddings)
        current_user_embeddings = torch.as_tensor(current_user_embeddings)
        previous_user_embeddings_all = torch.as_tensor(previous_user_embeddings_all)
        previous_item_embeddings_all = torch.as_tensor(previous_item_embeddings_all)

        current_user_embeddings = current_user_embeddings.view(805, -1)
        current_item_embeddings = current_item_embeddings.view(390, -1)
        previous_user_embeddings_all = previous_user_embeddings_all.view(805, -1)
        previous_item_embeddings_all = previous_item_embeddings_all.view(390, -1)

        current_user_embeddings = current_user_embeddings[self.userID]
        # print(previous_user_embeddings_all)
        previous_user_embeddings = previous_user_embeddings_all[self.userID]
        # print(previous_user_embeddings)

        norm_user_emb1 = F.normalize(current_user_embeddings, dim=0)
        norm_user_emb2 = F.normalize(previous_user_embeddings, dim=0)

        norm_all_user_emb = F.normalize(torch.as_tensor(previous_user_embeddings_all))
        # print(norm_all_user_emb)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()

        current_item_embeddings = current_item_embeddings[self.itemID]
        previous_item_embeddings = previous_item_embeddings_all[self.itemID]
        norm_item_emb1 = F.normalize(current_item_embeddings, dim=1)
        norm_item_emb2 = F.normalize(previous_item_embeddings, dim=1)

        norm_all_item_emb = F.normalize(torch.as_tensor(previous_item_embeddings_all))

        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss, norm_all_user_emb, norm_all_item_emb

    def run_kmeans(self, x, dim, _k):
        """Run K-means algorithm to get k clusters of the input tensor x
        """
        # Kmeans这个方法的输出是什么含义？
        kmeans = faiss.Kmeans(d=self.latent_dim, k=_k, gpu=True)
        x = np.reshape(x, [dim, self.latent_dim])
        kmeans.train(x)
        cluster_cents = kmeans.centroids

        _, I = kmeans.index.search(x, 1)

        # convert to cuda Tensors for broadcast
        centroids = torch.tensor(cluster_cents).to(device)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device)
        return centroids, node2cluster

    def e_step(self):
        user_embeddings = self.c_embedding_0.detach().cpu().numpy()
        item_embeddings = self.c_embedding_1.detach().cpu().numpy()
        # user_embeddings = self.c_embedding_0.weight.detach().cpu().numpy()
        # item_embeddings = self.c_embedding_1.weight.detach().cpu().numpy()
        self.user_centroids, self.user_2cluster = self.run_kmeans(user_embeddings, dim=805, _k=self.k)
        self.item_centroids, self.item_2cluster = self.run_kmeans(item_embeddings, dim=390, _k=self.k_1)

    def ProtoNCE_loss(self):
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
        lori_mp = -torch.log(matrix_mp2sc.mul(pos + 1e-13).sum(dim=-1)).mean()

        matrix_sc2mp = matrix_sc2mp / (torch.sum(matrix_sc2mp, dim=1).view(-1, 1) + 1e-8)
        lori_sc = -torch.log(matrix_sc2mp.mul(pos + 1e-13).sum(dim=-1)).mean()

        return self.lam * lori_mp + (1 - self.lam) * lori_sc, emb1, emb2

    def Same2Loss(self):

        same_loss_user, _, __ = self.SameContrast(self.ssl_u, self.nce_u, self.edge_u)
        same_loss_item, _, __ = self.SameContrast(self.ssl_i, self.nce_i, self.edge_i)
        user = torch.as_tensor(self.ssl_u + self.nce_u)
        item = torch.as_tensor(self.ssl_i + self.nce_i)
        return self.con_reg * torch.as_tensor(same_loss_user + same_loss_item), user, item

    def calculate_loss(self):
        ssl_loss, self.ssl_i, self.ssl_u = self.ssl_layer_loss()
        print(ssl_loss)
        nce_loss, self.nce_u, self.nce_i = self.ProtoNCE_loss()
        print(nce_loss)
        # contrast_loss = self.con_reg * torch.as_tensor(self.con_i + self.con_u, device=device).float()
        con_loss, self.con_u, self.con_i = self.Same2Loss()
        print(con_loss)

        return ssl_loss + nce_loss + con_loss
