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
from TransformerBlock import TransformerBlock
from sklearn.metrics import classification_report, accuracy_score


class MTL(torch.nn.Module):
    def __init__(self, config, in_feats, out_feats, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.out_feats = out_feats
        self.best_acc = 0
        self.patience = 0
        kernel = [5, 6, 7, 8]
        out_channels = 125
        maxlen1 = 230
        maxlen1b = 10
        self.maxlen1b = maxlen1b
        maxlen2 = 16
        self.maxlen2 = maxlen2
        V, embedding_dim = config['embedding_weights'].shape  # 词典的大小
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat_rumor1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        self.gat_rumor2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)

        self.gat_shared1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gat_shared2 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_shared1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_shared2 = GATConv(in_channels=in_feats, out_channels=out_feats)

        self.gat_stance1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        # self.gat_stance2 = GATConv(in_channels=in_feats * 3, out_channels=out_feats)
        self.gat_stance2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)

        self.n_heads = 16
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim,
                                           padding_idx=0, _weight=torch.from_numpy(config['embedding_weights']))

        self.self_attention = TransformerBlock(input_size=embedding_dim, n_heads=self.n_heads, attn_dropout=0)
        self.attention = AdditiveAttention(encoder_dim=out_channels * len(kernel),
                                           decoder_dim=out_channels * len(kernel))
        # self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1,
        #                         batch_first=True, bidirectional=True)
        # in_channels, out_channels, kernel_size
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=K) for K in kernel])
        # self.convs2 = nn.ModuleList(
        #     [nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=K) for K in kernel])
        # 池化层没有参数
        # 不妨试试平均池化
        # self.max_poolings1 = nn.ModuleList([nn.AvgPool1d(kernel_size=maxlen1 - K + 1) for K in kernel])
        self.max_poolings1 = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen1 - K + 1) for K in kernel])
        self.max_poolings1b = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen1b - K + 1) for K in kernel])
        self.max_poolings2 = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen2 - K + 1) for K in kernel])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # self.fc_rumor1 = nn.Linear(in_features=out_channels * len(kernel), out_features=300)
        self.fc_rumor1 = nn.Linear(in_features=out_feats * 2 + out_channels * len(kernel) * 2, out_features=300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=2)

        #
        text_fea_dim = out_channels * len(kernel)
        # self.fc_stance1 = nn.Linear(in_features=text_fea_dim*2, out_features=300)
        self.fc_stance1 = nn.Linear(in_features=out_feats + text_fea_dim*2, out_features=300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=3)
        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)

    def forward(self, batch):
        self.change_data_form(batch)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gat_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gat_shared1(batch.x2, batch.edge_index2)

        # 文本经过自注意力机制处理
        stance_text = batch.text2
        self.watch.append(stance_text[batch.root_index2])  # batch.root_index2是正确的
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
        batch.rootpost_index2 = root_indices2
        root_indices2 = root_indices2 + batch.root_indices

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
        y_pred2, y2, res2 = self.predict(X)
        acc2 = accuracy_score(y2, y_pred2)
        if acc2 > self.best_acc:
            self.best_acc = acc2
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y2, y_pred2, labels=[0, 1, 2], target_names=['a', 'b', 'c'], digits=5))
            print("Val　set acc:", acc2)
            print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc2, self.best_acc))
            self.patience += 1

        return acc2, res2

    def predict(self, data):
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred2 = []
        y2 = []
        for i, batch in enumerate(data):
            y2.extend(batch.y2.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            with torch.no_grad():
                # batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)

                # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
                output2 = self.forward(batch)
                predicted2 = torch.max(output2, dim=1)[1]
                predicted2 = predicted2.data.cpu().numpy().tolist()
                y_pred2 += predicted2
                # y_pred2 = None

        res2 = classification_report(y2,
                                     y_pred2,
                                     labels=[0, 1, 2],
                                     target_names=['a', 'b', 'c'],
                                     digits=5,
                                     output_dict=True)
        # res1 = None
        # res2 = None
        return y_pred2, y2, res2


class MTL_backup(torch.nn.Module):
    def __init__(self, config, in_feats, out_feats, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.out_feats = out_feats
        self.best_acc = 0
        self.patience = 0
        kernel = [5, 6, 7, 8]
        out_channels = 125
        maxlen1 = 230
        maxlen1b = 10
        self.maxlen1b = maxlen1b
        maxlen2 = 16
        self.maxlen2 = maxlen2
        V, embedding_dim = config['embedding_weights'].shape  # 词典的大小
        self.embedding_dim = embedding_dim
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.gat_rumor1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        self.gat_rumor2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)

        self.gat_shared1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        self.gat_shared2 = GATConv(in_channels=in_feats, out_channels=out_feats)

        self.gat_stance1 = GATConv(in_channels=in_feats, out_channels=out_feats)
        self.gat_stance2 = GATConv(in_channels=in_feats * 2 + embedding_dim, out_channels=out_feats)

        self.n_heads = 16
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim,
                                           padding_idx=0, _weight=torch.from_numpy(config['embedding_weights']))

        self.self_attention = TransformerBlock(input_size=embedding_dim, n_heads=self.n_heads, attn_dropout=0)
        self.attention = AdditiveAttention(encoder_dim=out_channels * len(kernel),
                                           decoder_dim=out_channels * len(kernel))
        # self.gru = torch.nn.GRU(input_size=embedding_dim, hidden_size=embedding_dim, num_layers=1,
        #                         batch_first=True, bidirectional=True)
        # in_channels, out_channels, kernel_size
        self.convs = nn.ModuleList(
            [nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=K) for K in kernel])
        # self.convs2 = nn.ModuleList(
        #     [nn.Conv1d(in_channels=embedding_dim, out_channels=out_channels, kernel_size=K) for K in kernel])
        # 池化层没有参数
        # 不妨试试平均池化
        # self.max_poolings1 = nn.ModuleList([nn.AvgPool1d(kernel_size=maxlen1 - K + 1) for K in kernel])
        self.max_poolings1 = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen1 - K + 1) for K in kernel])
        self.max_poolings1b = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen1b - K + 1) for K in kernel])
        self.max_poolings2 = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen2 - K + 1) for K in kernel])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # self.fc_rumor1 = nn.Linear(in_features=out_channels * len(kernel), out_features=300)
        self.fc_rumor1 = nn.Linear(in_features=out_feats * 2 + out_channels * len(kernel) * 2, out_features=300)
        self.fc_rumor2 = nn.Linear(in_features=300, out_features=2)

        self.fc_stance1 = nn.Linear(in_features=out_feats * 2 + out_channels * len(kernel), out_features=300)
        self.fc_stance2 = nn.Linear(in_features=300, out_features=3)
        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc_rumor1.weight)
        init.xavier_normal_(self.fc_rumor2.weight)

    def forward(self, batch):
        self.change_data_form(batch)
        ####################################
        # 谣言检测 第一层
        ####################################
        rumor_graph = self.gat_rumor1(batch.x, batch.edge_index)
        rumor_graphshared = self.gat_shared1(batch.x, batch.edge_index)

        # 文本经过自注意力机制处理
        rumor_post_text = batch.text1[batch.root_index]
        rumor_post_text = self.word_embedding(rumor_post_text)
        rumor_post_text = self.self_attention(rumor_post_text, rumor_post_text, rumor_post_text)

        rumor_comment_text = batch.text1[:, -self.maxlen1b:]
        rumor_comment_text = self.word_embedding(rumor_comment_text)
        rumor_comment_text = self.self_attention(rumor_comment_text, rumor_comment_text, rumor_comment_text)

        rumor_post_text_avg = torch.mean(rumor_post_text, dim=1)
        rumor_comment_text_avg = torch.mean(rumor_comment_text, dim=1)
        rumor_comment_text_avg[batch.root_index] = rumor_post_text_avg
        rumor_text = rumor_comment_text_avg

        rumor_concat_graph_graphshared_text = torch.cat([rumor_graph, rumor_graphshared, rumor_text], dim=1)
        rumor_graph = rumor_concat_graph_graphshared_text  # 第一层的输出

        ####################################
        # 谣言检测 第二层
        ####################################
        rumor_graph = self.gat_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_shared = self.gat_shared2(rumor_graphshared, batch.edge_index)

        # 谣言检测, 处理原文
        rumor_post_text = rumor_post_text.permute(0, 2, 1)
        rumor_post_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings1)):
            act = self.relu(Conv(rumor_post_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_post_text_conv.append(pool)

        # rumor_post_text_conv.append(rumor_concat_graph_graphshared[batch.root_index])
        rumor_post_feature = torch.cat(rumor_post_text_conv, dim=1)

        # 谣言检测, 处理评论
        rumor_comment_text = rumor_comment_text.permute(0, 2, 1)
        rumor_comment_text_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings1b)):
            act = self.relu(Conv(rumor_comment_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_comment_text_conv.append(pool)

        # rumor_post_text_conv.append(rumor_concat_graph_graphshared)
        rumor_text_feature = torch.cat(rumor_comment_text_conv, dim=1)
        rumor_text_feature[batch.root_index] = rumor_post_feature

        rumor_post_comment_att = []
        # additive attention
        for i, start in enumerate(batch.root_index):
            if i == len(batch.root_index) - 1:
                end = rumor_text_feature.shape[0]
            else:
                end = batch.root_index[i + 1]
            # 取出batch中第i+1个小图对应的文本特征
            post_comment = rumor_text_feature[start:end]
            post = post_comment[0, :]
            comment = post_comment[1:, :]
            res = self.attention(query=post, values=comment)
            rumor_post_comment_att.append(res)
        rumor_post_comment_att_feature = torch.stack(rumor_post_comment_att, dim=0)

        rumor_final_feature = torch.cat([rumor_graph[batch.root_index],
                                         rumor_graph_shared[batch.root_index],
                                         rumor_post_feature,
                                         rumor_post_comment_att_feature],
                                        dim=1)

        output1 = self.relu(self.fc_rumor1(rumor_final_feature))
        output1 = self.fc_rumor2(output1)

        ####################################
        # 立场检测 第一层
        ####################################
        stance_graph = self.gat_stance1(batch.x2, batch.edge_index2)
        stance_graphshared = self.gat_shared1(batch.x2, batch.edge_index2)

        # 文本经过自注意力机制处理
        stance_text = batch.text2
        self.watch.append(stance_text[batch.root_index2])  # batch.root_index2是正确的
        stance_text = self.word_embedding(stance_text)
        stance_text = self.self_attention(stance_text, stance_text, stance_text)

        stance_text_avg = torch.mean(stance_text, dim=1)

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
                                          stance_graph_shared[batch.root_index2],
                                          stance_text_feature[batch.root_index2]],
                                         dim=1)

        output2 = self.relu(self.fc_stance1(stance_final_feature))
        output2 = self.fc_stance2(output2)

        # output1 = None

        return output1, output2

    def forward_backup(self, batch):
        self.change_data_form(batch)
        batch.text1 = self.word_embedding(batch.text1)
        batch.text2 = self.word_embedding(batch.text2)

        # rumor detection
        graph_rumor_fea = self.gat_rumor(batch.x, batch.edge_index)

        # stance detection
        graph_stance_fea = self.gat_stance(batch.x2, batch.edge_index2)

        # text_rumor, _ = self.gru(batch.text1)
        # text_rumor = self.self_attention(text_rumor, text_rumor, text_rumor)
        root_text1 = batch.text1[batch.root_index]
        text_rumor = self.self_attention(root_text1, root_text1, root_text1)
        # text_rumor, _ = self.gru(text_rumor)
        text_rumor = text_rumor.permute(0, 2, 1)

        # conv_block_rumor = [graph_rumor_fea]
        conv_block_rumor = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings1)):
            act = self.relu(Conv(text_rumor))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            conv_block_rumor.append(pool)
        final_feature_rumor = torch.cat(conv_block_rumor, dim=1)
        # 这里出错了 batch.root_index越界?
        # final_feature_rumor = final_feature_rumor[batch.root_index]

        # text_stance, _ = self.gru(batch.text2)
        text_stance = self.self_attention(batch.text2, batch.text2, batch.text2)
        text_stance = text_stance.permute(0, 2, 1)

        conv_block_stance = [graph_stance_fea]
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings2)):
            act = self.relu(Conv(text_stance))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            conv_block_stance.append(pool)
        final_feature_stance = torch.cat(conv_block_stance, dim=1)
        final_feature_stance = final_feature_stance[batch.root_index2]

        # MLP rumor
        output1 = self.relu(self.fc_rumor1(final_feature_rumor))
        output1 = self.fc_rumor2(output1)

        # MLP stance
        output2 = self.relu(self.fc_stance1(final_feature_stance))
        output2 = self.fc_stance2(output2)

        return output1, output2

    def forward_onlyrumor(self, batch):
        self.change_data_form(batch)
        batch.text1 = self.word_embedding(batch.text1)

        root_text1 = batch.text1[batch.root_index]
        text_rumor = self.self_attention(root_text1, root_text1, root_text1)
        text_rumor = text_rumor.permute(0, 2, 1)

        conv_block_rumor = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings1)):
            act = self.relu(Conv(text_rumor))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            conv_block_rumor.append(pool)
        final_feature_rumor = torch.cat(conv_block_rumor, dim=1)

        # MLP rumor
        output1 = self.relu(self.fc_rumor1(final_feature_rumor))
        output1 = self.fc_rumor2(output1)

        output2 = None

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
        root_indices2 = root_indices2 + batch.root_indices

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
        # acc1 = None
        acc2 = accuracy_score(y2, y_pred2)
        if acc1 > self.best_acc:
            self.best_acc = acc1
            self.patience = 0
            torch.save(self.state_dict(), self.config['save_path'])
            print(classification_report(y1, y_pred1, target_names=['nonrumor', 'rumor'], digits=5))
            print("Val set acc:", acc1)
            print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc1, self.best_acc))
            self.patience += 1

        # if acc2 > self.best_acc:
        #     self.best_acc = acc2
        #     self.patience = 0
        #     torch.save(self.state_dict(), self.config['save_path'])
        #     print(classification_report(y2, y_pred2, labels=[0, 1, 2], target_names=['a', 'b', 'c'], digits=5))
        #     print("Val set acc:", acc2)
        #     print("Best val set acc:", self.best_acc)
        #     print("save model!!!")
        # else:
        #     print("本轮acc{}小于最优acc{}, 不保存模型".format(acc2, self.best_acc))
        #     self.patience += 1

        return acc1, acc2, res1, res2

    def predict(self, data):
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred1 = []
        y_pred2 = []
        y1 = []
        y2 = []
        for i, batch in enumerate(data):
            y1.extend(batch.y.data.cpu().numpy().tolist())
            y2.extend(batch.y2.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            with torch.no_grad():
                # batch_x_tid, batch_x_text = (item.cuda(device=self.device) for item in data)

                # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
                output1, output2 = self.forward(batch)
                predicted1 = torch.max(output1, dim=1)[1]
                predicted1 = predicted1.data.cpu().numpy().tolist()
                y_pred1 += predicted1
                # y_pred1 = None

                predicted2 = torch.max(output2, dim=1)[1]
                predicted2 = predicted2.data.cpu().numpy().tolist()
                y_pred2 += predicted2
                # y_pred2 = None

        res1 = classification_report(y1,
                                     y_pred1,
                                     labels=[0, 1],
                                     target_names=['non-rumor', 'rumor'],
                                     digits=5,
                                     output_dict=True)
        res2 = classification_report(y2,
                                     y_pred2,
                                     labels=[0, 1, 2],
                                     target_names=['a', 'b', 'c'],
                                     digits=5,
                                     output_dict=True)
        # res1 = None
        # res2 = None
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
