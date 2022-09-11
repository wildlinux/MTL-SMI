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

# nfeat == max len
nfeat = 120
random.seed(123)


###################
def load_data0():
    with open('dataset/weibo/adj_rumor.json', 'r', encoding='utf-8') as f:
        adj_rumor = json.load(f)
    adj_rumor, maprumor_mid_id, maprumor_id_mid = load_adj(adj_rumor)
    with open('dataset/weibo/label_rumor.json', 'r', encoding='utf-8') as f:
        label_rumor = json.load(f)
        maprumor_id_label = {}
        for mid, label in label_rumor.items():
            _id = maprumor_mid_id[mid]
            if label == 'non-rumor':
                maprumor_id_label[_id] = 1
            else:
                maprumor_id_label[_id] = 0
    ids = list(maprumor_id_label.keys())
    number = int(len(ids) * 0.8)
    random.shuffle(ids)
    rumor_train_ids = torch.LongTensor(ids[:number])
    rumor_test_ids = torch.LongTensor(ids[number:])
    rumor_labels = []
    for _id in ids:
        rumor_labels.append(maprumor_id_label[_id])
    rumor_labels = torch.LongTensor(rumor_labels)
    with open('dataset/weibo/text_rumor.json', 'r', encoding='utf-8') as f:
        text_rumor = json.load(f)
        text_rumor_final = {}
        for mid in text_rumor:
            if mid in maprumor_mid_id:
                text_rumor_final[mid] = text_rumor[mid]
        text_rumor = text_rumor_final
        n_text_rumor = len(text_rumor)
        features_rumor = np.zeros(shape=(n_text_rumor, nfeat))
        for mid in text_rumor:
            _id = maprumor_mid_id[mid]
            features_rumor[_id, :] = text_rumor[mid]
        features_rumor = torch.FloatTensor(features_rumor)

    with open('dataset/weibo_stance/adj_stance.json', 'r', encoding='utf-8') as f:
        adj_stance = json.load(f)
    adj_stance, mapstance_mid_id, mapstance_id_mid = load_adj(adj_stance)
    with open('dataset/weibo_stance/label_stance.json', 'r', encoding='utf-8') as f:
        label_stance = json.load(f)
        mapstance_id_label = {}
        for mid, label in label_stance.items():
            _id = mapstance_mid_id[mid]
            mapstance_id_label[_id] = int(label)
    ids = list(mapstance_id_label.keys())
    number = int(len(ids) * 0.8)
    random.shuffle(ids)
    stance_train_ids = torch.LongTensor(ids[:number])
    stance_test_ids = torch.LongTensor(ids[number:])
    stance_labels = []
    stance_train_labels = []
    for _id in ids:
        stance_labels.append(mapstance_id_label[_id])
    stance_labels = torch.LongTensor(stance_labels)

    with open('dataset/weibo_stance/text_stance.json', 'r', encoding='utf-8') as f:
        text_stance = json.load(f)
        text_stance_final = {}
        for mid in text_stance:
            if mid in mapstance_mid_id:
                text_stance_final[mid] = text_stance[mid]
        text_stance = text_stance_final
        n_text_stance = len(text_stance)
        features_stance = np.zeros(shape=(n_text_stance, nfeat))
        for mid in text_stance:
            _id = mapstance_mid_id[mid]
            features_stance[_id, :] = text_stance[mid]
        features_stance = torch.FloatTensor(features_stance)

    data = {'adj_rumor': adj_rumor, 'maprumor_id_label': maprumor_id_label, 'features_rumor': features_rumor,
            'rumor_train_ids': rumor_train_ids, 'rumor_test_ids': rumor_test_ids, 'rumor_labels': rumor_labels,
            'adj_stance': adj_stance, 'mapstance_id_label': mapstance_id_label, 'features_stance': features_stance,
            'stance_train_ids': stance_train_ids, 'stance_test_ids': stance_test_ids, 'stance_labels': stance_labels}
    return data


def load_adj(adj_json: dict):
    map_mid_id = {}
    map_id_mid = {}
    used_mid = []
    _id = 0
    for father, sons in adj_json.items():
        sons.append(father)
        for mid in sons:
            if mid in used_mid:
                continue
            used_mid.append(mid)
            map_mid_id[mid] = _id
            map_id_mid[_id] = mid
            _id += 1
    print('节点个数:{}'.format(len(map_mid_id)))
    N = len(map_mid_id)
    adj = np.zeros(shape=(N, N), dtype=np.float32)
    for father, sons in adj_json.items():
        index_father = map_mid_id[father]
        for son in sons:
            index_son = map_mid_id[son]
            adj[index_father, index_son] = 1
    adj = torch.FloatTensor(adj)
    return adj, map_mid_id, map_id_mid


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "2"
    # parse = argparse.ArgumentParser()
    # parse.add_argument("-d", "--dataset", help="dataset", type=str, required=True)
    # parse.add_argument("-l", "--labelrate", help="labeled data for train per class", type=int, required=True)
    # args = parse.parse_args()

    cuda = torch.cuda.is_available()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    seed = Config.seed
    np.random.seed(seed)
    torch.manual_seed(seed)

    data = load_data0()
    sadj = data['adj_rumor']
    fadj = data['adj_stance']
    features_rumor = data['features_rumor']
    features_stance = data['features_stance']
    maprumor_id_label = data['maprumor_id_label']
    mapstance_id_label = data['mapstance_id_label']
    rumor_labels = data['rumor_labels']
    stance_labels = data['stance_labels']
    rumor_train_ids = data['rumor_train_ids']
    rumor_test_ids = data['rumor_test_ids']
    stance_train_ids = data['stance_train_ids']
    stance_test_ids = data['stance_test_ids']

    # features, labels, idx_train, idx_test = load_data(Config)

    model = SFGCN(nfeat=nfeat,
                  nhid1=Config.nhid1,
                  nhid2=Config.nhid2,
                  nclass_rumor=2,
                  nclass_stance=3,
                  dropout=Config.dropout)

    if cuda:
        model.cuda()
        features_rumor = features_rumor.cuda()
        features_stance = features_stance.cuda()
        sadj = sadj.cuda()
        fadj = fadj.cuda()
        rumor_labels = rumor_labels.cuda()
        stance_labels = stance_labels.cuda()
        rumor_train_ids = rumor_train_ids.cuda()
        rumor_test_ids = rumor_test_ids.cuda()
        stance_train_ids = stance_train_ids.cuda()
        stance_test_ids = stance_test_ids.cuda()
    optimizer = optim.Adam(model.parameters(), lr=Config.lr)
    # optimizer1
    # optimizer2
    # 应该指定参数?  但是共享参数怎么办?
    # optimizer1 = optim.Adam(model.parameters(), lr=Config.lr)
    # optimizer2 = optim.Adam(model.parameters(), lr=Config.lr)



    def train(model, epochs):
        model.train()
        optimizer.zero_grad()
        output1, output2 = model(features_rumor, features_stance, sadj, fadj)
        # optimizer1.zero_grad()
        # xxx
        # loss_rumor = loss1 + alpha1 * loss_com
        # optimizer1.step()
        # print metrics

        # optimizer2.zero_grad()
        # xxx
        # loss_stance = loss2 + alpha2 * loss_com
        # optimizer2.step()
        # print metrics
        # label index is not consistent with rumor_labels
        loss_rumor = F.nll_loss(output1[rumor_train_ids], rumor_labels[:len(rumor_train_ids)])
        loss = loss_rumor
        # loss_class = F.nll_loss(output[idx_train], labels[idx_train])
        # loss_dep = (loss_dependence(emb1, com1, config.n) + loss_dependence(emb2, com2, config.n)) / 2
        # loss_com = common_loss(com1, com2)
        # loss = loss_class + config.beta * loss_dep + config.theta * loss_com
        # label index is not consistent with rumor_labels
        acc = accuracy(output1[rumor_train_ids], rumor_labels[:len(rumor_train_ids)])
        loss.backward()
        optimizer.step()
        print(acc)
        # acc_test, macro_f1, emb_test = main_test(model)
        # print('e:{}'.format(epochs),
        #       'ltr: {:.4f}'.format(loss.item()),
        #       'atr: {:.4f}'.format(acc.item()),
        #       'ate: {:.4f}'.format(acc_test.item()),
        #       'f1te:{:.4f}'.format(macro_f1.item()))
        # return loss.item(), acc_test.item(), macro_f1.item(), emb_test


    def main_test(model):
        model.eval()
        output1, output2 = model(features_rumor, features_stance, sadj, fadj)
        acc_test = accuracy(output1[rumor_test_ids], rumor_labels[rumor_test_ids])
        label_max = []
        for idx in rumor_test_ids:
            label_max.append(torch.argmax(output1[idx]).item())
        labelcpu = rumor_labels[rumor_test_ids].data.cpu()
        macro_f1 = f1_score(labelcpu, label_max, average='macro')
        return acc_test, macro_f1, emb


    acc_max = 0
    f1_max = 0
    epoch_max = 0
    for epoch in range(Config.epochs):
        # loss, acc_test, macro_f1, emb = train(model, epoch)
        train(model, epoch)
    #     if acc_test >= acc_max:
    #         acc_max = acc_test
    #         f1_max = macro_f1
    #         epoch_max = epoch
    # print('epoch:{}'.format(epoch_max),
    #       'acc_max: {:.4f}'.format(acc_max),
    #       'f1_max: {:.4f}'.format(f1_max))
