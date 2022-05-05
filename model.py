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
    def __init__(self, ss_num, hh_num, sh_num, embedding_dim, batchSize, drop):
        super(KDHR, self).__init__()
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
        self.ssl_temp = 0.05
        self.ssl_reg = 1e-6
        self.alpha = 1.5
        self.latent_dim = 64
        self.k = 20
        self.k_1 = 9
        self.proto_reg = 8e-8
        self.device = torch.device("cuda", 0)
        self.device_0 = torch.device("cpu")

        self.ssl_u, self.ssl_i = [], []
        self.nce_u, self.nce_i = [], []

        self.user_centroids = None
        self.user_2cluster = None
        self.item_centroids = None
        self.item_2cluster = None

        self.userID = torch.linspace(0, 804, 805).long()
        self.itemID = torch.linspace(0, 389, 390).long()

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

    def forward(self, x_SH, edge_index_SH, x_SS, edge_index_SS, x_HH, edge_index_HH, prescription):
        # S-H图搭建
        # 第一层
        x_SH1 = self.SH_embedding(x_SH.long())
        # x_SH1 = torch.tensor(torch.nn.Embedding(1195, 64).view(1195, 1, 64)

        # 这里的x_SH2和下边的x_SH22有区别，目前不知道什么原因
        x_SH2 = self.convSH_TostudyS_1(x_SH1.float(), edge_index_SH)
        # 第二层
        x_SH6 = self.convSH_TostudyS_2(x_SH2, edge_index_SH)
        # x_SH6 = x_SH6.view(-1, 256)
        # 第三层
        # x_SH7 = self.convSH_TostudyS_3(x_SH6, edge_index_SH)

        # x_SH9 = (x_SH1 + x_SH2 + x_SH6 ) / 3.0
        # x_SH9 = self.SH_mlp_1(x_SH9)
        # x_SH9 = x_SH9.view(1195, -1)
        # x_SH9 = self.SH_bn_1(x_SH9)
        # x_SH9 = self.SH_tanh_1(x_SH9)

        # SH H
        # 0: 草药 1: 症状
        x_SH11 = self.SH_embedding(x_SH.long())
        x_SH22 = self.convSH_TostudyS_1_h(x_SH11.float(), edge_index_SH)
        # 第二层
        x_SH66 = self.convSH_TostudyS_2_h(x_SH22, edge_index_SH)
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
        # n_i:805 n_u:390
        es = torch.tensor(s_u + n_u)#+ s_lossemb
        eh = torch.tensor(s_i + n_i)#+ h_lossemb
        # es = torch.tensor(self.ssl_i + self.nce_i) + s_lossemb
        # eh = torch.tensor(self.ssl_u + self.nce_u) + h_lossemb
        # cat
        pass

        # sum
        # es = x_SH9[:390] + x_ss1# ss图view  # 1195,1,64  390,1,64
        # eh = x_SH99[390:] + x_hh1 # hh图view 805*dim
        # cat
        # es = torch.cat((x_SH9[:390], x_ss1), dim=-1)
        # eh = torch.cat((x_SH99[390:], x_hh1), dim=-1)

        # SI 集成多个症状为一个症状表示 batch*390 390*dim => batch*dim
        es = es.view(390, -1)
        e_synd = torch.mm(prescription, es)  # prescription * es
        # batch*1
        preSum = prescription.sum(dim=1).view(-1, 1)
        # batch*dim
        e_synd_norm = e_synd / preSum
        e_synd_norm = self.mlp(e_synd_norm)
        #e_synd_norm = e_synd_norm.view(-1, 256)
        e_synd_norm = self.SI_bn(e_synd_norm)
        e_synd_norm = self.relu(e_synd_norm)  # batch*dim
        # batch*dim dim*805 => batch*805
        eh = eh.view(805, -1)
        pre = torch.mm(e_synd_norm, eh.t())

        return pre

    def ssl_layer_loss(self):  # KDHR的0层和N层
        # self.c_embedding_0 = torch.tensor(self.c_embedding_0)
        # self.p_embedding_0 = torch.tensor(self.p_embedding_0)

        current_user_embeddings, current_item_embeddings = self.c_embedding_0, self.c_embedding_1
        previous_user_embeddings_all, previous_item_embeddings_all = self.p_embedding_0, self.p_embedding_1

        current_item_embeddings = torch.tensor(current_item_embeddings)
        current_user_embeddings = torch.tensor(current_user_embeddings)
        previous_user_embeddings_all = torch.tensor(previous_user_embeddings_all)
        previous_item_embeddings_all = torch.tensor(previous_item_embeddings_all)

        current_user_embeddings = current_user_embeddings.view(805, -1)
        current_item_embeddings = current_item_embeddings.view(390, -1)
        previous_user_embeddings_all = previous_user_embeddings_all.view(805, -1)
        previous_item_embeddings_all = previous_item_embeddings_all.view(390, -1)



        current_user_embeddings = current_user_embeddings[self.userID]
        #print(previous_user_embeddings_all)
        previous_user_embeddings = previous_user_embeddings_all[self.userID]
        #print(previous_user_embeddings)

        norm_user_emb1 = F.normalize(current_user_embeddings, dim=0)
        norm_user_emb2 = F.normalize(previous_user_embeddings, dim=0)

        norm_all_user_emb = F.normalize(torch.tensor(previous_user_embeddings_all))
        #print(norm_all_user_emb)
        pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
        ttl_score_user = torch.matmul(norm_user_emb1, norm_all_user_emb.transpose(0, 1))
        pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
        ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)

        ssl_loss_user = -torch.log(pos_score_user / ttl_score_user).sum()


        current_item_embeddings = current_item_embeddings[self.itemID]
        previous_item_embeddings = previous_item_embeddings_all[self.itemID]
        norm_item_emb1 = F.normalize(current_item_embeddings, dim=1)
        norm_item_emb2 = F.normalize(previous_item_embeddings, dim=1)

        norm_all_item_emb = F.normalize(previous_item_embeddings_all)

        pos_score_item = torch.mul(norm_item_emb1, norm_item_emb2).sum(dim=1)
        ttl_score_item = torch.matmul(norm_item_emb1, norm_all_item_emb.transpose(0, 1))
        pos_score_item = torch.exp(pos_score_item / self.ssl_temp)
        ttl_score_item = torch.exp(ttl_score_item / self.ssl_temp).sum(dim=1)

        ssl_loss_item = -torch.log(pos_score_item / ttl_score_item).sum()

        ssl_loss = self.ssl_reg * (ssl_loss_user + self.alpha * ssl_loss_item)
        return ssl_loss, norm_all_user_emb, norm_all_item_emb

    ## faiss库安装未成功，原始安装应为GPU版的
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
        centroids = torch.tensor(cluster_cents).to(self.device_0)
        centroids = F.normalize(centroids, p=2, dim=1)

        node2cluster = torch.LongTensor(I).squeeze().to(self.device_0)
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
        user_embeddings_all = torch.tensor(user_embeddings_all)
        item_embeddings_all = torch.tensor(item_embeddings_all)

        norm_user_embeddings = F.normalize(user_embeddings_all, dim=0)
        norm_user_embeddings = norm_user_embeddings.view(805, -1)
        self.e_step()
        user2cluster = self.user_2cluster[self.userID]  # [B,]

        user2centroids = self.user_centroids[user2cluster]  # [B, e]
        pos_score_user = torch.mul(norm_user_embeddings, user2centroids).sum(dim=1)
        pos_score_user = torch.exp(pos_score_user/ self.ssl_temp)
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

    def calculate_loss(self):
        # clear the storage variable when training
        # if self.restore_user_e is not None or self.restore_item_e is not None:
        #     self.restore_user_e, self.restore_item_e = None, None

        # current_user_embeddings, current_item_embeddings = self.c_embedding_0, self.c_embedding_1
        # previous_user_embeddings_all, previous_item_embeddings_all = self.p_embedding_0, self.p_embedding_1
        # pos_item = interaction[self.ITEM_ID]
        # neg_item = interaction[self.NEG_ITEM_ID]

        # user_all_embeddings, item_all_embeddings, embeddings_list = self.forward()

        # center_embedding = embeddings_list[0]
        # context_embedding = embeddings_list[self.hyper_layers * 2]
        #
        # ssl_loss = self.ssl_layer_loss(context_embedding, center_embedding, user, item)
        # proto_loss = self.ProtoNCE_loss(center_embedding, user, item)
        #
        # u_embeddings = user_all_embeddings[user]
        # i_embedding = item_all_embeddings[item]
        # pos_embeddings = item_all_embeddings[pos_item]
        # neg_embeddings = item_all_embeddings[neg_item]

        # calculate BPR Loss
        # pos_scores = torch.mul(u_embeddings, pos_embeddings).sum(dim=1)
        # neg_scores = torch.mul(u_embeddings, neg_embeddings).sum(dim=1)

        # mf_loss = self.mf_loss(pos_scores, neg_scores)
        #
        # u_ego_embeddings = self.user_embedding(user)
        # pos_ego_embeddings = self.item_embedding(pos_item)
        # neg_ego_embeddings = self.item_embedding(neg_item)
        #
        # reg_loss = self.reg_loss(u_ego_embeddings, pos_ego_embeddings, neg_ego_embeddings)

        # return mf_loss + self.reg_weight * reg_loss, ssl_loss, proto_loss
        ssl_loss, self.ssl_u, self.ssl_i = self.ssl_layer_loss()
        nce_loss, self.nce_u, self.nce_i = self.ProtoNCE_loss()

        return ssl_loss + nce_loss

    # def same2ssl(self, x, index):
    #     ssl_loss = 0
    #     for i, j in enumerate(index):
    #         # x和index是通过Data方法转换后的形式
    #         norm_user_emb1, norm_user_emb2 = x[i], x[j]
    #
    #         pos_score_user = torch.mul(norm_user_emb1, norm_user_emb2).sum(dim=1)
    #         ttl_score_user = torch.matmul(norm_user_emb1, self.p_embedding_0[805].transpose(0, 1))
    #         pos_score_user = torch.exp(pos_score_user / self.ssl_temp)
    #         ttl_score_user = torch.exp(ttl_score_user / self.ssl_temp).sum(dim=1)
    #
    #         ssl_loss = -torch.log(pos_score_user / ttl_score_user).sum()
    #
    #     return ssl_loss, x
