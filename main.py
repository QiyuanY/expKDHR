#!/usr/bin/env python
# -*- coding:utf-8 -*-
# Author: Rao Yulong

from utils import *
from model import *
# from model_compara import Compare
# from model_SMGCN import SMGCN
import sys
import os
import optuna
import parameter
import torch
from pytorchtools import EarlyStopping
from load import table2mat
import time
from load_data import DataLoad

# import Lr_auto

seed = 2021512
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(device)


class Logger(object):
    def __init__(self, filename="Default.log"):
        self.terminal = sys.stdout
        self.log = open(filename, "w")

    def write(self, message):
        self.terminal.write(message)
        self.log.write(message)

    def flush(self):
        pass


path = os.path.abspath(os.path.dirname(__file__))
type = sys.getfilesystemencoding()
sys.stdout = Logger('khdr.txt')

para = parameter.para(lr=0.056, rec=1e-4, drop=0.0, batchSize=8192, epoch=500, dev_ratio=0.2, test_ratio=0.2)
ld = DataLoad(para)
# para = parameter.para(lr=0.05, rec=3e-3, drop=0.0, batchSize=8192, epoch=200, dev_ratio=0.2, test_ratio=0.2)
train_dataset, dev_dataset, test_dataset = ld.GetDataset()
ss_edge_adj, hh_edge_adj, sh_data = ld.GetIndex()
x_train, x_dev, x_test = ld.GetSet()
# print(Lr_auto.params)
print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
print(" dropout: ", para.drop, " batchsize: ",
      para.batchSize, " epoch: ", para.epoch, " dev_ratio: ", para.dev_ratio, " test_ratio: ", para.test_ratio)

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=para.batchSize)
dev_loader = torch.utils.data.DataLoader(dev_dataset, batch_size=para.batchSize)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=para.batchSize)

criterion = torch.nn.BCEWithLogitsLoss(reduction="mean").to(device)

early_stopping = EarlyStopping(patience=7, verbose=True)

epsilon = 1e-13

hh_edge_adj = torch.as_tensor(hh_edge_adj).to(device)
ss_edge_adj = torch.as_tensor(ss_edge_adj).to(device)
sh_data.edge_index = sh_data.edge_index.to(device)
model = KDHR(390, 805, 1195, 64, ss_edge_adj, hh_edge_adj, para.batchSize, para.drop).to(device)


def objective(trial):
    # params = {
    #     # 'n_layers_rnn': trial.suggest_int('n_layers_rnn', 1, 4),
    #     # 'n_units_rnn': trial.set_user_attr('n_units_rnn', 256),
    #     # 'activation': trial.suggest_categorical('activation', ['relu', 'tanh']),
    #     # 'dropout': trial.suggest_uniform('dropout', 0.1, 0.7),
    #     'lr': trial.suggest_loguniform('lr', 0.045, 0.065),
    #     'rec': trial.suggest_loguniform('rec', 6e-4, 1e-3)
    # }
    optimizer = torch.optim.Adam(model.parameters(), lr=para.lr, weight_decay=para.rec)

    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=7, gamma=0.8)
    for epoch in range(para.epoch):

        model.train()
        running_loss = 0.0
        for i, (sid, hid) in enumerate(train_loader):
            # sid, hid = sid.to(device), hid.to(device)
            # sid, hid = sid.float().to(device), hid.float().to(device)
            optimizer.zero_grad()
            # batch*805 概率矩阵
            outputs = model(sh_data.edge_index, sid)
            # outputs = model(sh_data.x, sh_data_adj, ss_data.x, ss_data_adj, hh_data.x, hh_data_adj, sid)
            loss = criterion(outputs, hid) + model.calculate_loss()
            loss = loss.to(device)
            loss.backward()
            optimizer.step()
            # print(scheduler.get_lr())
            running_loss += loss.item()
        # print train loss per every epoch
        print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(train_loader))
        # print('[Epoch {}]train_loss: '.format(epoch + 1), running_loss / len(x_train))
        # loss_list.append(running_loss / len(train_loader))

        model.eval()
        dev_loss = 0

        dev_p5 = 0
        dev_p10 = 0
        dev_p20 = 0

        dev_r5 = 0
        dev_r10 = 0
        dev_r20 = 0

        dev_f1_5 = 0
        dev_f1_10 = 0
        dev_f1_20 = 0
        for tsid, thid in dev_loader:
            # tsid, thid = tsid.float().to(device), thid.float().to(device)
            # batch*805 概率矩阵
            outputs = model(sh_data.edge_index, tsid)
            # outputs = model(sh_edge_index, ss_co, hh_co, tsid)

            # outputs = model(sh_data.x, sh_data_adj, ss_data.x, ss_data_adj, hh_data.x, hh_data_adj, tsid)
            dev_loss += (criterion(outputs, thid) + model.calculate_loss()).item()

            # thid batch*805
            for i, hid in enumerate(thid):
                # trueLabel = []  # 对应存在草药的索引
                # for idx, val in enumerate(hid):  # 获得thid中值为一的索引
                #     if val == 1:
                #         trueLabel.append(idx)
                trueLabel = (hid == 1).nonzero().flatten()
                top5 = torch.topk(outputs[i], 5)[1]  # 预测值前5索引
                count = 0
                for m in top5:
                    if m in trueLabel:
                        count += 1
                dev_p5 += count / 5
                dev_r5 += count / len(trueLabel)
                # dev_f1_5 += 2*(count / 5)*(count / len(trueLabel)) / ((count / 5) + (count / len(trueLabel)) + epsilon)

                top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
                count = 0
                for m in top10:
                    if m in trueLabel:
                        count += 1
                dev_p10 += count / 10
                dev_r10 += count / len(trueLabel)
                # dev_f1_10 += 2 * (count / 10) * (count / len(trueLabel)) / ((count / 10) + (count / len(trueLabel)) + epsilon)

                top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
                count = 0
                for m in top20:
                    if m in trueLabel:
                        count += 1
                dev_p20 += count / 20
                dev_r20 += count / len(trueLabel)
                # dev_f1_20 += 2 * (count / 20) * (count / len(trueLabel)) / ((count / 20) + (count / len(trueLabel)) + epsilon)

        scheduler.step()

        print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(dev_loader))
        # print('[Epoch {}]dev_loss: '.format(epoch + 1), dev_loss / len(x_dev))
        print('d:p5-10-20:', dev_p5 / len(x_dev), dev_p10 / len(x_dev), dev_p20 / len(x_dev))
        print('d:r5-10-20:', dev_r5 / len(x_dev), dev_r10 / len(x_dev), dev_r20 / len(x_dev))
        # print('f1_5-10-20: ', dev_f1_5 / len(x_dev), dev_f1_10 / len(x_dev), dev_f1_20 / len(x_dev))
        print('d:f1_5-10-20: ',
              2 * (dev_p5 / len(x_dev)) * (dev_r5 / len(x_dev)) / (
                      (dev_p5 / len(x_dev)) + (dev_r5 / len(x_dev)) + epsilon),
              2 * (dev_p10 / len(x_dev)) * (dev_r10 / len(x_dev)) / (
                      (dev_p10 / len(x_dev)) + (dev_r10 / len(x_dev)) + epsilon),
              2 * (dev_p20 / len(x_dev)) * (dev_r20 / len(x_dev)) / (
                      (dev_p20 / len(x_dev)) + (dev_r20 / len(x_dev)) + epsilon))

        early_stopping(dev_loss / len(dev_loader), model)
        if early_stopping.early_stop:
            print("Early stopping")
            break

    print(time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime()))
    # 获得 early stopping 时的模型参数
    model.load_state_dict(torch.load('checkpoint.pt'))

    model.eval()
    test_loss = 0

    test_p5 = 0
    test_p10 = 0
    test_p20 = 0

    test_r5 = 0
    test_r10 = 0
    test_r20 = 0

    test_f1_5 = 0
    test_f1_10 = 0
    test_f1_20 = 0

    for tsid, thid in test_loader:
        # tsid, thid = tsid.float(), thid.float()
        # batch*805 概率矩阵
        outputs = model(sh_data.edge_index, tsid)
        # outputs = model(sh_edge_index, ss_co, hh_co, tsid)

        test_loss += criterion(outputs, thid).item()
        # thid batch*805
        for i, hid in enumerate(thid):
            # trueLabel = []  # 对应存在草药的索引
            # for idx, val in enumerate(hid):  # 获得thid中值为一的索引
            #     if val == 1:
            #         trueLabel.append(idx)
            trueLabel = (hid == 1).nonzero().flatten()
            top5 = torch.topk(outputs[i], 5)[1]  # 预测值前5索引
            count = 0
            for m in top5:
                if m in trueLabel:
                    count += 1
            test_p5 += count / 5
            test_r5 += count / len(trueLabel)

            top10 = torch.topk(outputs[i], 10)[1]  # 预测值前10索引
            count = 0
            for m in top10:
                if m in trueLabel:
                    count += 1
            test_p10 += count / 10
            test_r10 += count / len(trueLabel)

            top20 = torch.topk(outputs[i], 20)[1]  # 预测值前20索引
            count = 0
            for m in top20:
                if m in trueLabel:
                    count += 1
            test_p20 += count / 20
            test_r20 += count / len(trueLabel)

    print("----------------------------------------------------------------------------------------------------")

    print('test_loss: ', test_loss / len(test_loader))

    print('p5-10-20:', test_p5 / len(x_test), test_p10 / len(x_test), test_p20 / len(x_test))
    print('r5-10-20:', test_r5 / len(x_test), test_r10 / len(x_test), test_r20 / len(x_test))

    score_f1 = []

    print('f1_5-10-20: ',
          2 * (test_p5 / len(x_test)) * (test_r5 / len(x_test)) / ((test_p5 / len(x_test)) + (test_r5 / len(x_test))),
          2 * (test_p10 / len(x_test)) * (test_r10 / len(x_test)) / (
                  (test_p10 / len(x_test)) + (test_r10 / len(x_test))),
          2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / (
                  (test_p20 / len(x_test)) + (test_r20 / len(x_test))))

    score_f1 = 2 * (test_p20 / len(x_test)) * (test_r20 / len(x_test)) / (
            (test_p20 / len(x_test)) + (test_r20 / len(x_test)))
    # print(model.g)
    # DrawPic(model.g)
    return score_f1


# study = optuna.create_study(direction='maximize')
# study.optimize(objective, n_trials=100)
#
# trial_ = study.best_trial
# print(f'min loss: {trial_.value}')
# print(f'best params: {trial_.params}')  # 输出最优结果的模型超参数

objective(1)
