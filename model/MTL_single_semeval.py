import sys, os

sys.path.append(os.getcwd())
import time
import torch as th
import torch
import torch.nn as nn
import torch.nn.init as init
from torch_scatter import scatter_mean
import torch.nn.functional as F
import numpy as np
from torch_geometric.data import DataLoader
from tqdm import tqdm
from torch_geometric.nn import GCNConv, GATConv
import copy
import json
from .TransformerBlock_sign import TransformerBlock
from sklearn.metrics import classification_report, accuracy_score

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)


class MTL(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=300, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0
        # kernel = [4, 5, 6, 7]
        # out_channels = 120
        # maxlen1 = 20
        # maxlen1b = 10
        # self.maxlen1b = maxlen1b
        # maxlen2 = 16
        # self.maxlen2 = maxlen2
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # self.gat_rumor1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_rumor2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)
        #
        # self.gat_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_shared2 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        #
        # self.gat_stance1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_stance2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)

        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                               text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                               is_layer_norm=config['self_att_layer_norm'])

        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(out_channels * kernel_num, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])

        self.fc_tmp1 = nn.Linear(in_features=300, out_features=150)
        self.fc_tmp2 = nn.Linear(in_features=150, out_features=config['num_classes'])

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, batch):
        # self.change_data_form(batch)
        # 文本经过自注意力机制处理
        stance_text = batch.text
        # self.watch.append(stance_text[batch.root_index])  # batch.root_index2是正确的
        # 对谁预测就处理谁
        stance_text = stance_text[batch.root_index]
        self.watch.append(stance_text)
        stance_text = self.word_embedding(stance_text)
        X_text_pos, X_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([X_text_pos, X_text_neg], dim=2)

        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_text_feature = torch.cat(stance_text_conv, dim=1)
        features = stance_text_feature

        output = self.relu(self.fc1(features))
        output = self.fc2(output)

        return output

    def evaluate(self, X):
        y_pred, y, res = self.predict(X)
        acc = accuracy_score(y, y_pred)
        if acc > self.best_acc:
            self.best_acc = acc
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print("Val　set acc:", acc)
            print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        '''
sum_ = 0
for x in self.watch:
    sum_ += x.sum()
print(sum_)


aa = [x.data.cpu().numpy().tolist() for x in self.watch]
with open('../MTGNN/vs_testdata.json', 'w') as f:
    json.dump(aa, f)

with open('vs_testdata.json', 'r') as f:
    aa = json.load(f)
bb = [torch.LongTensor(x).cuda() for x in aa]


for i in range(len(bb)):
    if 0 != ((bb[i] != self.watch[i]).sum()):
        print(i)

cc = bb[4]
dd = self.watch[4]
for i in range(len(cc)):
    if 0 != ((cc[i] != dd[i]).sum()):
        print(i)


string = ''
for x in self.watch:
    res = x.data.cpu().numpy().tolist()
    res = [str(x) for x in res]
    string += ''.join(res)
import hashlib
res = hashlib.md5(string.encode('utf-8')).digest()
print(res)

        '''
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []



        total = len(data)
        for i, batch in enumerate(data):
            # if i == 0:
            #     self.watch = []
            # if i == 2:
            #     print(i)
            # if i == 4:
            #     print(i)
            if i == total - 1:
                print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            with torch.no_grad():
                batch.x = batch.x.cuda(device=self.device)
                batch.text = batch.text.cuda(device=self.device)
                batch.root_index = batch.root_index.cuda(device=self.device)
                batch.edge_index = batch.edge_index.cuda(device=self.device)
                batch.y = batch.y.cuda(device=self.device)
                # batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)

                # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
                output = self.forward(batch)
                predicted = torch.max(output, dim=1)[1]
                predicted = predicted.data.cpu().numpy().tolist()
                y_pred += predicted
                # y_pred2 = None
        res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res


def forward_single_semeval_backup(self, batch):
    # self.change_data_form(batch)

    ####################################
    # 立场检测 第一层
    ####################################
    stance_graph = self.gat_stance1(batch.x, batch.edge_index)
    stance_graphshared = self.gat_shared1(batch.x, batch.edge_index)

    # 文本经过自注意力机制处理
    stance_text = batch.text
    # self.watch.append(stance_text[batch.root_index])  # batch.root_index2是正确的
    stance_text = self.word_embedding(stance_text)
    stance_text = self.self_attention(stance_text, stance_text, stance_text)

    stance_text_avg = torch.mean(stance_text, dim=1)

    # stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_graph], dim=1)
    stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
    stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

    ####################################
    # 立场检测 第二层
    ####################################
    stance_graph = self.gat_stance2(stance_graph, batch.edge_index2)
    stance_graph_shared = self.gat_shared2(stance_graphshared, batch.edge_index2)

    # 谣言检测, 处理原文
    stance_text = stance_text.permute(0, 2, 1)
    stance_text_conv = []
    for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings2)):
        act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
        pool = max_pooling(act)  # 128 * 100 * 1
        pool = torch.squeeze(pool)
        stance_text_conv.append(pool)

    stance_text_feature = torch.cat(stance_text_conv, dim=1)
    stance_final_feature = torch.cat([stance_graph[batch.root_index2],
                                      stance_text_feature[batch.root_index2],
                                      stance_text_feature[batch.rootpost_index2]
                                      ],
                                     dim=1)
    # stance_final_feature = stance_text_feature[batch.rootpost_index2]

    output2 = self.relu(self.fc_stance1(stance_final_feature))
    output2 = self.fc_stance2(output2)

    return output2


class AdditiveAttention(torch.nn.Module):
    def __init__(self, encoder_dim=300, decoder_dim=300):
        super().__init__()

        self.encoder_dim = encoder_dim
        self.decoder_dim = decoder_dim
        self.v = torch.nn.Parameter(torch.rand(self.decoder_dim))
        self.W_1 = torch.nn.Linear(self.decoder_dim, self.decoder_dim)
        self.W_2 = torch.nn.Linear(self.encoder_dim, self.decoder_dim)

    # query is a decoder hidden state h
    # values is a matrix of encoder hidden states s
    def forward(self, query, values):
        '''

        :param query: [decoder_dim]
        :param values: [seq_length, encoder_dim]
        :return:
        '''
        weights = self._get_weights(query, values)  # [seq_length]
        weights = torch.nn.functional.softmax(weights, dim=0)
        # res就是values加权求和的结果
        res = weights @ values  # [encoder_dim]
        return res

    # _get_weights就是打分函数 score(h,s) =  v tanh(W1h + W2s)
    def _get_weights(self, query, values):
        '''

        :param query: [decoder_dim]
        :param values: [seq_length, encoder_dim]
        :return:
        '''
        query = query.repeat(values.size(0), 1)  # [seq_length, decoder_dim]
        weights = self.W_1(query) + self.W_2(values)  # [seq_length, decoder_dim]
        return torch.tanh(weights) @ self.v  # [seq_length]


class Attention(nn.Module):
    def __init__(self, in_size, hidden_size=16):
        super(Attention, self).__init__()

        self.project = nn.Sequential(
            nn.Linear(in_size, hidden_size),
            nn.Tanh(),
            nn.Linear(hidden_size, 1, bias=False)
        )

    def forward(self, z):
        w = self.project(z)
        beta = torch.softmax(w, dim=1)
        return (beta * z).sum(1), beta
