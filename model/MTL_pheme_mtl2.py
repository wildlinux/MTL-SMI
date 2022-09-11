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
from .TransformerBlock import TransformerBlock_Original
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertModel, BertConfig

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# 不考虑交互
class MTL1(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # additive_att的输出维度也是out_channels * kernel_num
        self.fc_rumor1 = nn.Linear(out_channels * kernel_num + out_feats * 2, 300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=3)

        self.fc_stance1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # 使用了正负att, 所以是300*2
        self.gnn_rumor2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_shared2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_stance1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_stance2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        # self.attention = AdditiveAttention(encoder_dim=out_channels * kernel_num,
        #                                    decoder_dim=out_channels * kernel_num)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)
        init.xavier_normal_(self.fc_stance1.weight)
        init.xavier_normal_(self.fc_stance2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)


        # rumor_text[0](n,L,768)
        rumor_text = self.bert(batch.input_ids1, attention_mask=batch.attention_mask1)
        # (n,L,768)
        rumor_text = rumor_text[0]
        # (n,L,dim)
        rumor_text = self.fc(rumor_text)

        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # (bs,L,2d)
        # rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        # rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_concat_graph_graphshared_text = rumor_graph
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        # 只考虑post
        rumor_text = rumor_text[batch.root_index]
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # stance_text = batch.text2
        # stance_text = self.word_embedding(stance_text)

        # stance_text[0](n,L,768)
        stance_text = self.bert(batch.input_ids2, attention_mask=batch.attention_mask2)
        # (n,L,768)
        stance_text = stance_text[0]
        # (n,L,dim)
        stance_text = self.fc(stance_text)


        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        # stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        # stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_concat_graph_graphshared_text = stance_graph
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_reserve_additiveATT(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_highgpu(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # 只考虑post和待分类的comment即可
        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_feature_text = torch.cat(stance_text_conv, dim=1)
        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_text[batch.root_index2],
            stance_feature_text[batch.post_index2]],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    # 手动将graph2拼接起来
    def change_data_form(self, batch):
        edge_indices2 = batch.edge_indices2
        graph2_edge_num = batch.graph2_edge_num
        graph2_node_num = batch.graph2_node_num
        # 转置成2*n的形式
        edge_indices2 = edge_indices2.transpose(1, 0)
        # 每个graph2的索引不再从0开始, 需要考虑之前的所有graph2
        # delta_indices2记录每个原始索引需要增加的大小
        i = 0
        delta_indices2 = []
        for index, edge_num in enumerate(graph2_edge_num):
            delta_indices2.extend([i] * edge_num)
            i = i + int(graph2_node_num[index])
        #############这里有错
        root_indices2 = list(set(delta_indices2))
        root_indices2.sort()
        root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        # 考虑立场分类中的post index
        batch.post_index2 = root_indices2

        root_indices2 = root_indices2 + batch.root_indices2

        # root_indices2.sort()
        # 每个graph2的根节点索引
        # root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        #############这里有错

        # 让每个原始索引加上需要增加的值
        delta_indices2 = torch.LongTensor(delta_indices2).to(self.device)
        row = edge_indices2[0] + delta_indices2
        col = edge_indices2[1] + delta_indices2
        # 组合成batch后的边的索引
        # edge_indices22 = torch.LongTensor(torch.stack((row, col), 0))
        edge_indices22 = torch.stack((row, col), 0)
        # 更新batch中的数据
        batch.root_index2 = root_indices2
        batch.edge_index2 = edge_indices22

    def evaluate(self, X):
        y_pred1, y_pred2, y1, y2, res1, res2 = self.predict(X)
        acc1 = accuracy_score(y1, y_pred1)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y1, y_pred1, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        return acc1, res1

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())

            output1, output2 = self.forward(batch)
            predicted1 = torch.max(output1, dim=1)[1]
            predicted1 = predicted1.data.cpu().numpy().tolist()
            y_pred1 += predicted1

            predicted2 = torch.max(output2, dim=1)[1]
            predicted2 = predicted2.data.cpu().numpy().tolist()
            y_pred2 += predicted2

            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res1 = classification_report(y1, y_pred1, digits=5, output_dict=True)
        print("rumor acc:{}    f1:{}".format(res1['accuracy'], res1['macro avg']['f1-score']))

        res2 = classification_report(y2, y_pred2, digits=5, output_dict=True)
        print("stance acc:{}    f1:{}".format(res2['accuracy'], res2['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred1, y_pred2, y1, y2, res1, res2

# 只有文本共享层进行交互
class MTL2(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # additive_att的输出维度也是out_channels * kernel_num
        self.fc_rumor1 = nn.Linear(out_channels * kernel_num + out_feats * 2, 300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=3)

        self.fc_stance1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # 使用了正负att, 所以是300*2
        self.gnn_rumor2 = GCNConv(in_channels=out_feats + 300 * 2, out_channels=out_feats)

        self.gnn_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_shared2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_stance1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_stance2 = GCNConv(in_channels=out_feats + 300 * 2, out_channels=out_feats)

        # self.attention = AdditiveAttention(encoder_dim=out_channels * kernel_num,
        #                                    decoder_dim=out_channels * kernel_num)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)
        init.xavier_normal_(self.fc_stance1.weight)
        init.xavier_normal_(self.fc_stance2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)


        # rumor_text[0](n,L,768)
        rumor_text = self.bert(batch.input_ids1, attention_mask=batch.attention_mask1)
        # (n,L,768)
        rumor_text = rumor_text[0]
        # (n,L,dim)
        rumor_text = self.fc(rumor_text)

        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # (bs,L,2d)
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        # 只考虑post
        rumor_text = rumor_text[batch.root_index]
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # stance_text = batch.text2
        # stance_text = self.word_embedding(stance_text)

        # stance_text[0](n,L,768)
        stance_text = self.bert(batch.input_ids2, attention_mask=batch.attention_mask2)
        # (n,L,768)
        stance_text = stance_text[0]
        # (n,L,dim)
        stance_text = self.fc(stance_text)


        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_reserve_additiveATT(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_highgpu(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # 只考虑post和待分类的comment即可
        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_feature_text = torch.cat(stance_text_conv, dim=1)
        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_text[batch.root_index2],
            stance_feature_text[batch.post_index2]],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    # 手动将graph2拼接起来
    def change_data_form(self, batch):
        edge_indices2 = batch.edge_indices2
        graph2_edge_num = batch.graph2_edge_num
        graph2_node_num = batch.graph2_node_num
        # 转置成2*n的形式
        edge_indices2 = edge_indices2.transpose(1, 0)
        # 每个graph2的索引不再从0开始, 需要考虑之前的所有graph2
        # delta_indices2记录每个原始索引需要增加的大小
        i = 0
        delta_indices2 = []
        for index, edge_num in enumerate(graph2_edge_num):
            delta_indices2.extend([i] * edge_num)
            i = i + int(graph2_node_num[index])
        #############这里有错
        root_indices2 = list(set(delta_indices2))
        root_indices2.sort()
        root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        # 考虑立场分类中的post index
        batch.post_index2 = root_indices2

        root_indices2 = root_indices2 + batch.root_indices2

        # root_indices2.sort()
        # 每个graph2的根节点索引
        # root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        #############这里有错

        # 让每个原始索引加上需要增加的值
        delta_indices2 = torch.LongTensor(delta_indices2).to(self.device)
        row = edge_indices2[0] + delta_indices2
        col = edge_indices2[1] + delta_indices2
        # 组合成batch后的边的索引
        # edge_indices22 = torch.LongTensor(torch.stack((row, col), 0))
        edge_indices22 = torch.stack((row, col), 0)
        # 更新batch中的数据
        batch.root_index2 = root_indices2
        batch.edge_index2 = edge_indices22

    def evaluate(self, X):
        y_pred1, y_pred2, y1, y2, res1, res2 = self.predict(X)
        acc1 = accuracy_score(y1, y_pred1)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y1, y_pred1, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        return acc1, res1

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())

            output1, output2 = self.forward(batch)
            predicted1 = torch.max(output1, dim=1)[1]
            predicted1 = predicted1.data.cpu().numpy().tolist()
            y_pred1 += predicted1

            predicted2 = torch.max(output2, dim=1)[1]
            predicted2 = predicted2.data.cpu().numpy().tolist()
            y_pred2 += predicted2

            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res1 = classification_report(y1, y_pred1, digits=5, output_dict=True)
        print("rumor acc:{}    f1:{}".format(res1['accuracy'], res1['macro avg']['f1-score']))

        res2 = classification_report(y2, y_pred2, digits=5, output_dict=True)
        print("stance acc:{}    f1:{}".format(res2['accuracy'], res2['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred1, y_pred2, y1, y2, res1, res2

# 只有图共享层进行交互
class MTL3(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # additive_att的输出维度也是out_channels * kernel_num
        self.fc_rumor1 = nn.Linear(out_channels * kernel_num + out_feats * 2, 300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=3)

        self.fc_stance1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # 使用了正负att, 所以是300*2
        self.gnn_rumor2 = GCNConv(in_channels=out_feats * 2, out_channels=out_feats)

        self.gnn_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_shared2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_stance1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_stance2 = GCNConv(in_channels=out_feats * 2 , out_channels=out_feats)

        # self.attention = AdditiveAttention(encoder_dim=out_channels * kernel_num,
        #                                    decoder_dim=out_channels * kernel_num)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)
        init.xavier_normal_(self.fc_stance1.weight)
        init.xavier_normal_(self.fc_stance2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)


        # rumor_text[0](n,L,768)
        rumor_text = self.bert(batch.input_ids1, attention_mask=batch.attention_mask1)
        # (n,L,768)
        rumor_text = rumor_text[0]
        # (n,L,dim)
        rumor_text = self.fc(rumor_text)

        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # (bs,L,2d)
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        # 只考虑post
        rumor_text = rumor_text[batch.root_index]
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # stance_text = batch.text2
        # stance_text = self.word_embedding(stance_text)

        # stance_text[0](n,L,768)
        stance_text = self.bert(batch.input_ids2, attention_mask=batch.attention_mask2)
        # (n,L,768)
        stance_text = stance_text[0]
        # (n,L,dim)
        stance_text = self.fc(stance_text)


        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_reserve_additiveATT(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_highgpu(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # 只考虑post和待分类的comment即可
        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_feature_text = torch.cat(stance_text_conv, dim=1)
        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_text[batch.root_index2],
            stance_feature_text[batch.post_index2]],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    # 手动将graph2拼接起来
    def change_data_form(self, batch):
        edge_indices2 = batch.edge_indices2
        graph2_edge_num = batch.graph2_edge_num
        graph2_node_num = batch.graph2_node_num
        # 转置成2*n的形式
        edge_indices2 = edge_indices2.transpose(1, 0)
        # 每个graph2的索引不再从0开始, 需要考虑之前的所有graph2
        # delta_indices2记录每个原始索引需要增加的大小
        i = 0
        delta_indices2 = []
        for index, edge_num in enumerate(graph2_edge_num):
            delta_indices2.extend([i] * edge_num)
            i = i + int(graph2_node_num[index])
        #############这里有错
        root_indices2 = list(set(delta_indices2))
        root_indices2.sort()
        root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        # 考虑立场分类中的post index
        batch.post_index2 = root_indices2

        root_indices2 = root_indices2 + batch.root_indices2

        # root_indices2.sort()
        # 每个graph2的根节点索引
        # root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        #############这里有错

        # 让每个原始索引加上需要增加的值
        delta_indices2 = torch.LongTensor(delta_indices2).to(self.device)
        row = edge_indices2[0] + delta_indices2
        col = edge_indices2[1] + delta_indices2
        # 组合成batch后的边的索引
        # edge_indices22 = torch.LongTensor(torch.stack((row, col), 0))
        edge_indices22 = torch.stack((row, col), 0)
        # 更新batch中的数据
        batch.root_index2 = root_indices2
        batch.edge_index2 = edge_indices22

    def evaluate(self, X):
        y_pred1, y_pred2, y1, y2, res1, res2 = self.predict(X)
        acc1 = accuracy_score(y1, y_pred1)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y1, y_pred1, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        return acc1, res1

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())

            output1, output2 = self.forward(batch)
            predicted1 = torch.max(output1, dim=1)[1]
            predicted1 = predicted1.data.cpu().numpy().tolist()
            y_pred1 += predicted1

            predicted2 = torch.max(output2, dim=1)[1]
            predicted2 = predicted2.data.cpu().numpy().tolist()
            y_pred2 += predicted2

            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res1 = classification_report(y1, y_pred1, digits=5, output_dict=True)
        print("rumor acc:{}    f1:{}".format(res1['accuracy'], res1['macro avg']['f1-score']))

        res2 = classification_report(y2, y_pred2, digits=5, output_dict=True)
        print("stance acc:{}    f1:{}".format(res2['accuracy'], res2['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred1, y_pred2, y1, y2, res1, res2

# 不使用signed attention
class MTL5(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mh_attention = TransformerBlock_Original(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # additive_att的输出维度也是out_channels * kernel_num
        self.fc_rumor1 = nn.Linear(out_channels * kernel_num + out_feats * 2, 300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=3)

        self.fc_stance1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # 使用了正负att, 所以是300*2
        self.gnn_rumor2 = GCNConv(in_channels=out_feats * 2 + 300 , out_channels=out_feats)

        self.gnn_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_shared2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_stance1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_stance2 = GCNConv(in_channels=out_feats * 2 + 300 , out_channels=out_feats)

        # self.attention = AdditiveAttention(encoder_dim=out_channels * kernel_num,
        #                                    decoder_dim=out_channels * kernel_num)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)
        init.xavier_normal_(self.fc_stance1.weight)
        init.xavier_normal_(self.fc_stance2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)


        # rumor_text[0](n,L,768)
        rumor_text = self.bert(batch.input_ids1, attention_mask=batch.attention_mask1)
        # (n,L,768)
        rumor_text = rumor_text[0]
        # (n,L,dim)
        rumor_text = self.fc(rumor_text)

        rumor_text = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        # 只考虑post
        rumor_text = rumor_text[batch.root_index]
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # stance_text = batch.text2
        # stance_text = self.word_embedding(stance_text)

        # stance_text[0](n,L,768)
        stance_text = self.bert(batch.input_ids2, attention_mask=batch.attention_mask2)
        # (n,L,768)
        stance_text = stance_text[0]
        # (n,L,dim)
        stance_text = self.fc(stance_text)


        stance_text = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_reserve_additiveATT(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_highgpu(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # 只考虑post和待分类的comment即可
        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_feature_text = torch.cat(stance_text_conv, dim=1)
        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_text[batch.root_index2],
            stance_feature_text[batch.post_index2]],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    # 手动将graph2拼接起来
    def change_data_form(self, batch):
        edge_indices2 = batch.edge_indices2
        graph2_edge_num = batch.graph2_edge_num
        graph2_node_num = batch.graph2_node_num
        # 转置成2*n的形式
        edge_indices2 = edge_indices2.transpose(1, 0)
        # 每个graph2的索引不再从0开始, 需要考虑之前的所有graph2
        # delta_indices2记录每个原始索引需要增加的大小
        i = 0
        delta_indices2 = []
        for index, edge_num in enumerate(graph2_edge_num):
            delta_indices2.extend([i] * edge_num)
            i = i + int(graph2_node_num[index])
        #############这里有错
        root_indices2 = list(set(delta_indices2))
        root_indices2.sort()
        root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        # 考虑立场分类中的post index
        batch.post_index2 = root_indices2

        root_indices2 = root_indices2 + batch.root_indices2

        # root_indices2.sort()
        # 每个graph2的根节点索引
        # root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        #############这里有错

        # 让每个原始索引加上需要增加的值
        delta_indices2 = torch.LongTensor(delta_indices2).to(self.device)
        row = edge_indices2[0] + delta_indices2
        col = edge_indices2[1] + delta_indices2
        # 组合成batch后的边的索引
        # edge_indices22 = torch.LongTensor(torch.stack((row, col), 0))
        edge_indices22 = torch.stack((row, col), 0)
        # 更新batch中的数据
        batch.root_index2 = root_indices2
        batch.edge_index2 = edge_indices22

    def evaluate(self, X):
        y_pred1, y_pred2, y1, y2, res1, res2 = self.predict(X)
        acc1 = accuracy_score(y1, y_pred1)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y1, y_pred1, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        return acc1, res1

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())

            output1, output2 = self.forward(batch)
            predicted1 = torch.max(output1, dim=1)[1]
            predicted1 = predicted1.data.cpu().numpy().tolist()
            y_pred1 += predicted1

            predicted2 = torch.max(output2, dim=1)[1]
            predicted2 = predicted2.data.cpu().numpy().tolist()
            y_pred2 += predicted2

            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res1 = classification_report(y1, y_pred1, digits=5, output_dict=True)
        print("rumor acc:{}    f1:{}".format(res1['accuracy'], res1['macro avg']['f1-score']))

        res2 = classification_report(y2, y_pred2, digits=5, output_dict=True)
        print("stance acc:{}    f1:{}".format(res2['accuracy'], res2['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred1, y_pred2, y1, y2, res1, res2

# 完整结构 _Complete
class MTL(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))

        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])
        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        # additive_att的输出维度也是out_channels * kernel_num
        self.fc_rumor1 = nn.Linear(out_channels * kernel_num + out_feats * 2, 300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=3)

        self.fc_stance1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # 使用了正负att, 所以是300*2
        self.gnn_rumor2 = GCNConv(in_channels=out_feats * 2 + 300 * 2, out_channels=out_feats)

        self.gnn_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_shared2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_stance1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_stance2 = GCNConv(in_channels=out_feats * 2 + 300 * 2, out_channels=out_feats)

        # self.attention = AdditiveAttention(encoder_dim=out_channels * kernel_num,
        #                                    decoder_dim=out_channels * kernel_num)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)
        init.xavier_normal_(self.fc_stance1.weight)
        init.xavier_normal_(self.fc_stance2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)


        # rumor_text[0](n,L,768)
        rumor_text = self.bert(batch.input_ids1, attention_mask=batch.attention_mask1)
        # (n,L,768)
        rumor_text = rumor_text[0]
        # (n,L,dim)
        rumor_text = self.fc(rumor_text)

        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # (bs,L,2d)
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        # 只考虑post
        rumor_text = rumor_text[batch.root_index]
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # stance_text = batch.text2
        # stance_text = self.word_embedding(stance_text)

        # stance_text[0](n,L,768)
        stance_text = self.bert(batch.input_ids2, attention_mask=batch.attention_mask2)
        # (n,L,768)
        stance_text = stance_text[0]
        # (n,L,dim)
        stance_text = self.fc(stance_text)


        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_reserve_additiveATT(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文和对应的comment, 不需要考虑所有的comment, 太占显存!
        # 处理comment
        stance_posttext = stance_text[batch.post_index2]
        stance_posttext = stance_posttext.permute(0, 2, 1)
        stance_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_posttext_conv.append(pool)
        stance_feature_posttext = torch.cat(stance_posttext_conv, dim=1)
        # 处理post
        stance_commenttext = stance_text[batch.root_index2]
        stance_commenttext = stance_commenttext.permute(0, 2, 1)
        stance_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_commenttext_conv.append(pool)
        stance_feature_commenttext = torch.cat(stance_commenttext_conv, dim=1)

        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_posttext,
            stance_feature_commenttext],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_highgpu(self, batch):
        # 转换graph2中的数据
        self.change_data_form(batch)

        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gnn_shared1(batch.x, batch.edge_index)

        ############################################################
        # 不使用 mean pooling
        ############################################################
        # rumor_graph = rumor_graph[batch.root_index]

        ############################################################
        # 使用 mean pooling
        # mean pooling(慢,因为有for循环)   # 其实不妨使用超级节点(快,不需要for循环)  实际上感觉超级节点更慢
        ############################################################
        # list_pool = []
        # last = len(batch.root_index) - 1
        # for i in range(len(batch.root_index)):
        #     if i == last:
        #         left = batch.root_index[i]
        #         cur = torch.mean(rumor_graph[left:], dim=0)
        #     else:
        #         left = batch.root_index[i]
        #         right = batch.root_index[i+1]
        #         cur = torch.mean(rumor_graph[left:right], dim=0)
        #     list_pool.append(cur)
        # rumor_graph = torch.stack(list_pool, dim=0)

        # 文本经过自注意力机制处理

        rumor_text = batch.text1
        rumor_text = self.word_embedding(rumor_text)
        rumor_text_pos, rumor_text_neg = self.mh_attention(rumor_text, rumor_text, rumor_text)
        rumor_text = torch.cat([rumor_text_pos, rumor_text_neg], dim=2)  # n,L,2d
        rumor_text_avg = torch.mean(rumor_text, dim=1)  # n,2d

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text_avg], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gnn_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测
        rumor_text = rumor_text.permute(0, 2, 1)  # n,2d,L
        rumor_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_text_conv.append(pool)
        # (batch, filter_num*out_channels)
        rumor_feature_text = torch.cat(rumor_text_conv, dim=1)

        # additive attention
        rumor_post_comment_att = []
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_feature_text.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_feature_text[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        # (batch, filter_num*out_channels)
        rumor_feature_post_comment_att = torch.stack(rumor_post_comment_att, dim=0)

        rumor_feature = torch.cat([rumor_graph[batch.root_index],
                                   rumor_graph_shared[batch.root_index],
                                   rumor_feature_text[batch.root_index],
                                   rumor_feature_post_comment_att]
                                  , dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gnn_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gnn_shared1(batch.x2, batch.edge_index2)

        # 只考虑post和待分类的comment即可
        stance_text = batch.text2
        stance_text = self.word_embedding(stance_text)
        stance_text_pos, stance_text_neg = self.mh_attention(stance_text, stance_text, stance_text)
        stance_text = torch.cat([stance_text_pos, stance_text_neg], dim=2)  # n,L,2d
        stance_text_avg = torch.mean(stance_text, dim=1)  # n,2d

        stance_concat_graph_graphshared_text = torch.cat([stance_graph, stance_graphshared, stance_text_avg], dim=1)
        stance_graph = stance_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 立场检测 第二层
        ####################################
        stance_graph = self.gnn_stance2(stance_graph, batch.edge_index2)
        stance_graph_shared = self.gnn_shared2(stance_graphshared, batch.edge_index2)
        # 谣言检测, 处理原文
        stance_text = stance_text.permute(0, 2, 1)
        stance_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(stance_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            stance_text_conv.append(pool)

        stance_feature_text = torch.cat(stance_text_conv, dim=1)
        # 64, 1600
        stance_final_feature = torch.cat([
            stance_graph[batch.root_index2],
            stance_graph_shared[batch.root_index2],
            stance_feature_text[batch.root_index2],
            stance_feature_text[batch.post_index2]],
            dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        return output1, output2

    # 手动将graph2拼接起来
    def change_data_form(self, batch):
        edge_indices2 = batch.edge_indices2
        graph2_edge_num = batch.graph2_edge_num
        graph2_node_num = batch.graph2_node_num
        # 转置成2*n的形式
        edge_indices2 = edge_indices2.transpose(1, 0)
        # 每个graph2的索引不再从0开始, 需要考虑之前的所有graph2
        # delta_indices2记录每个原始索引需要增加的大小
        i = 0
        delta_indices2 = []
        for index, edge_num in enumerate(graph2_edge_num):
            delta_indices2.extend([i] * edge_num)
            i = i + int(graph2_node_num[index])
        #############这里有错
        root_indices2 = list(set(delta_indices2))
        root_indices2.sort()
        root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        # 考虑立场分类中的post index
        batch.post_index2 = root_indices2

        root_indices2 = root_indices2 + batch.root_indices2

        # root_indices2.sort()
        # 每个graph2的根节点索引
        # root_indices2 = torch.LongTensor(root_indices2).to(self.device)
        #############这里有错

        # 让每个原始索引加上需要增加的值
        delta_indices2 = torch.LongTensor(delta_indices2).to(self.device)
        row = edge_indices2[0] + delta_indices2
        col = edge_indices2[1] + delta_indices2
        # 组合成batch后的边的索引
        # edge_indices22 = torch.LongTensor(torch.stack((row, col), 0))
        edge_indices22 = torch.stack((row, col), 0)
        # 更新batch中的数据
        batch.root_index2 = root_indices2
        batch.edge_index2 = edge_indices22

    def evaluate(self, X):
        y_pred1, y_pred2, y1, y2, res1, res2 = self.predict(X)
        acc1 = accuracy_score(y1, y_pred1)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y1, y_pred1, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        return acc1, res1

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())

            output1, output2 = self.forward(batch)
            predicted1 = torch.max(output1, dim=1)[1]
            predicted1 = predicted1.data.cpu().numpy().tolist()
            y_pred1 += predicted1

            predicted2 = torch.max(output2, dim=1)[1]
            predicted2 = predicted2.data.cpu().numpy().tolist()
            y_pred2 += predicted2

            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res1 = classification_report(y1, y_pred1, digits=5, output_dict=True)
        print("rumor acc:{}    f1:{}".format(res1['accuracy'], res1['macro avg']['f1-score']))

        res2 = classification_report(y2, y_pred2, digits=5, output_dict=True)
        print("stance acc:{}    f1:{}".format(res2['accuracy'], res2['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred1, y_pred2, y1, y2, res1, res2


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
