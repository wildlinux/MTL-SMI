from __future__ import division
from __future__ import print_function
import torch
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from utils import *
from models import SFGCN
import numpy
from sklearn.metrics import f1_score
import os
import torch.nn as nn
import argparse
from config import Config
import json
import random
from model.MTL import MTL
from sklearn.metrics import classification_report, accuracy_score

# nfeat == max len
nfeat = 120
random.seed(123)

from dataset import MTLDataset, MTLDataset_fortest
from torch_geometric.data import DataLoader


###################


def load_data():
    pass


if __name__ == "__main__":
    data_path_train = 'dataset/MTL_data_train_weibo.pkl'
    data_path_test = 'dataset/MTL_data_test_weibo.pkl'
    data_train = MTLDataset(path=data_path_train)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=False)
    data_test = MTLDataset(path=data_path_test)
    data_test_loader = DataLoader(data_test, batch_size=64, shuffle=False)
    # data_test_loader = DataLoader(data_test, batch_size=len(data_test), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    model_path = 'dataset/weibo_model'
    os.makedirs(model_path, exist_ok=True)
    config['save_path'] = os.path.join(model_path, 'BestModel')
    word_embeddings = np.load('dataset/word_embeddings.npy')
    config['embedding_weights'] = word_embeddings

    model = MTL(config, 300, 300).to(device)
    # attack和defense得用两个分类损失 ? 不是的, 一个就可以; 有权重需求的话可以用两个, 由于数据集分布相对均匀, 所以就不考虑权重问题了
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    epoches = 6
    total = len(data_train_loader)
    for epoch in range(epoches):
        model.train()
        for i, batch in enumerate(data_train_loader):
            output1, output2, grad = model.forward(batch)
            model.zero_grad()
            loss1 = loss(output1, batch.y)
            loss2 = loss(output2, batch.y2)
            lossloss = loss1 + loss2
            lossloss.backward()
            optimizer.step()
            corrects1 = (torch.max(output1, 1)[1].view(batch.y.size()).data == batch.y.data).sum()
            accuracy1 = 100 * corrects1 / len(batch.y)
            print(
                'Batch[{}/{}] - rumor loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                         loss1.item(),
                                                                                         accuracy1,
                                                                                         corrects1,
                                                                                         batch.y.size(0)))
            corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
            accuracy2 = 100 * corrects2 / len(batch.y2)
            print(
                'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                         loss2.item(),
                                                                                         accuracy2,
                                                                                         corrects2,
                                                                                         batch.y2.size(0)))

            output1, output2, grad = model.forward_adv(batch, grad['stance_final_feature'])
            model.zero_grad()
            loss1 = loss(output1, batch.y)
            loss2 = loss(output2, batch.y2)
            lossloss = loss1 + loss2
            lossloss.backward()
            optimizer.step()
            corrects1 = (torch.max(output1, 1)[1].view(batch.y.size()).data == batch.y.data).sum()
            accuracy1 = 100 * corrects1 / len(batch.y)
            print(
                'Batch[{}/{}] - rumor_adv loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                         loss1.item(),
                                                                                         accuracy1,
                                                                                         corrects1,
                                                                                         batch.y.size(0)))
            corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
            accuracy2 = 100 * corrects2 / len(batch.y2)
            print(
                'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                         loss2.item(),
                                                                                         accuracy2,
                                                                                         corrects2,
                                                                                         batch.y2.size(0)))


        print("epoch{}在测试集上的效果".format(epoch + 1))
        acc1, acc2, res1, res2 = model.evaluate(data_test_loader)
        print('acc1', acc1, 'acc2', acc2)













    # for epoch in range(epoches):
    #     model.train()
    #     grad = 0
    #     for i, batch in enumerate(data_train_loader):
    #         output1, output2, grad = model.forward(batch, grad)
    #         model.zero_grad()
    #         loss1 = loss(output1, batch.y)
    #         loss2 = loss(output2, batch.y2)
    #         # lossloss = loss2
    #         lossloss = loss1 + loss2
    #         lossloss.backward()
    #         # loss1.backward()
    #         # loss2.backward()
    #         optimizer.step()
    #         corrects1 = (torch.max(output1, 1)[1].view(batch.y.size()).data == batch.y.data).sum()
    #         accuracy1 = 100 * corrects1 / len(batch.y)
    #         print(
    #             'Batch[{}/{}] - rumor loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
    #                                                                                      loss1.item(),
    #                                                                                      accuracy1,
    #                                                                                      corrects1,
    #                                                                                      batch.y.size(0)))
    #         corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
    #         accuracy2 = 100 * corrects2 / len(batch.y2)
    #         print(
    #             'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
    #                                                                                      loss2.item(),
    #                                                                                      accuracy2,
    #                                                                                      corrects2,
    #                                                                                      batch.y2.size(0)))
    #     print("epoch{}在测试集上的效果".format(epoch + 1))
    #     acc1, acc2, res1, res2 = model.evaluate(data_test_loader)
    #     print('acc1', acc1, 'acc2', acc2)
    #     # print('acc1', acc1, 'acc2', acc2)
    #     # print('res1', res1, 'res2', res2)

    model.load_state_dict(torch.load(config['save_path']))
    y_pred1, y_pred2, y1, y2, res_rumor, res_stance = model.predict(data_test_loader)
    for k, v in res_rumor.items():
        print(k, v)
    print("==============================================================")
    print("谣言检测准确率:{:.4f}".format(res_rumor['accuracy']))
    print("==============================================================")

    # res_stance = classification_report(y2, y_pred2, target_names=['1', '2', '3'], digits=5, output_dict=True)
    for k, v in res_stance.items():
        print(k, v)
    print("==============================================================")
    print("立场分类准确率:{:.4f}".format(res_stance['accuracy']))
    print("==============================================================")