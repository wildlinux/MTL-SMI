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
# from .TransformerBlock import TransformerBlock
from sklearn.metrics import classification_report, accuracy_score
from transformers import BertModel, BertConfig

torch.backends.cudnn.benchmark = True
torch.manual_seed(0)
torch.cuda.manual_seed_all(0)

# _BERT
class MTL_BERT(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear(768, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=4)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, batch):
        ############################################################
        # 处理comment文本
        ############################################################

        # rumor_commenttext[0](n,L,768)
        rumor_commenttext = self.bert(batch.input_ids1[batch.root_index],
                                      attention_mask=batch.attention_mask1[batch.root_index])
        # (n,L,768)
        rumor_commenttext = rumor_commenttext[1]
        # rumor_commenttext = rumor_commenttext[0][:, 0, :]
        features = rumor_commenttext

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
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y, y_pred, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []
        total = len(data)
        for i, batch in enumerate(data):
            # if i == total - 1:
            #     print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            # with torch.no_grad():
            #     batch.x = batch.x.cuda(device=self.device)
            #     batch.text = batch.text.cuda(device=self.device)
            #     batch.root_index = batch.root_index.cuda(device=self.device)
            #     batch.edge_index = batch.edge_index.cuda(device=self.device)
            #     batch.y = batch.y.cuda(device=self.device)

            # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
            output = self.forward(batch)
            predicted = torch.max(output, dim=1)[1]
            predicted = predicted.data.cpu().numpy().tolist()
            y_pred += predicted
            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        print("acc:{}    f1:{}".format(res['accuracy'], res['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res


# _TEXTONLYBERT
class MTL_TEXTONLYBERT(torch.nn.Module):
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

        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # 正负attention
        # self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
        #                                      text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
        #                                      is_layer_norm=config['self_att_layer_norm'])

        # 普通attention
        # self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
        #                                      n_heads=config['n_heads'], attn_dropout=0,
        #                                      is_layer_norm=config['self_att_layer_norm'])

        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))

        # out_channels = config['nb_filters']
        # kernel_num = len(config['kernel_sizes'])
        # # 正负attention
        # self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        # # 普通attention
        # # self.convs = nn.ModuleList([nn.Conv1d(300, out_channels, kernel_size=K) for K in config['kernel_sizes']])
        # self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        # for GCN
        self.fc1 = nn.Linear(300 * 2 + out_feats * 2, 300)
        # for GAT
        # self.fc1 = nn.Linear(out_channels * kernel_num * 2 + out_feats * 5 * 2, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_rumor2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        ############################################################
        # 处理图
        ############################################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_comment = rumor_graph[batch.root_index]
        rumor_graph_post = rumor_graph[batch.post_index]

        ############################################################
        # 处理comment文本
        ############################################################

        # rumor_commenttext[0](n,L,768)
        rumor_commenttext = self.bert(batch.input_ids1[batch.root_index],
                                      attention_mask=batch.attention_mask1[batch.root_index])
        # (n,768)
        rumor_commenttext = rumor_commenttext[1]
        # (n,dim)
        rumor_commenttext = self.fc(rumor_commenttext)

        ############################################################
        # 处理post文本
        ############################################################
        # rumor_posttext[0](n,L,768)
        rumor_posttext = self.bert(batch.input_ids1[batch.post_index],
                                   attention_mask=batch.attention_mask1[batch.post_index])
        # (n,768)
        rumor_posttext = rumor_posttext[1]
        # (n,dim)
        rumor_posttext = self.fc(rumor_posttext)

        # (n, out_channels * kernel_num * 2 + out_feats * 2)
        features = torch.cat([rumor_commenttext, rumor_posttext, rumor_graph_comment, rumor_graph_post], dim=1)

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
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y, y_pred, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []
        total = len(data)
        for i, batch in enumerate(data):
            # if i == total - 1:
            #     print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            # with torch.no_grad():
            #     batch.x = batch.x.cuda(device=self.device)
            #     batch.text = batch.text.cuda(device=self.device)
            #     batch.root_index = batch.root_index.cuda(device=self.device)
            #     batch.edge_index = batch.edge_index.cuda(device=self.device)
            #     batch.y = batch.y.cuda(device=self.device)

            # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
            output = self.forward(batch)
            predicted = torch.max(output, dim=1)[1]
            predicted = predicted.data.cpu().numpy().tolist()
            y_pred += predicted
            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        print("acc:{}    f1:{}".format(res['accuracy'], res['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res

# _GCN
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

        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
                                           _weight=torch.from_numpy(embedding_weights))


        out_channels = config['nb_filters']
        kernel_num = len(config['kernel_sizes'])


        # 正负attention
        self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
                                             text_len=maxlen, n_heads=config['n_heads'], attn_dropout=0,
                                             is_layer_norm=config['self_att_layer_norm'])

        self.convs = nn.ModuleList([nn.Conv1d(300 * 2, out_channels, kernel_size=K) for K in config['kernel_sizes']])


        # 普通attention
        # self.mh_attention = TransformerBlock(input_size=300, d_k=config['self_att_dim'], d_v=config['self_att_dim'],
        #                                      n_heads=config['n_heads'], attn_dropout=0,
        #                                      is_layer_norm=config['self_att_layer_norm'])
        #
        # self.convs = nn.ModuleList([nn.Conv1d(300, out_channels, kernel_size=K) for K in config['kernel_sizes']])


        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])
        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()
        self.fc1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=4)
        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_rumor2 = GCNConv(in_channels=out_feats, out_channels=out_feats)
        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        ############################################################
        # 处理图
        ############################################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_comment = rumor_graph[batch.root_index]
        rumor_graph_post = rumor_graph[batch.post_index]

        ############################################################
        # 处理comment文本
        ############################################################

        # rumor_commenttext[0](n,L,768)
        rumor_commenttext = self.bert(batch.input_ids1[batch.root_index],
                                      attention_mask=batch.attention_mask1[batch.root_index])
        rumor_commenttext = rumor_commenttext[0]
        # (n,L,dim)
        rumor_commenttext = self.fc(rumor_commenttext)

        # 正负attention   (n,L,dim)
        rumor_commenttext_pos, rumor_commenttext_neg = self.mh_attention(rumor_commenttext, rumor_commenttext,
                                                                         rumor_commenttext)
        rumor_commenttext = torch.cat([rumor_commenttext_pos, rumor_commenttext_neg], dim=2) # (n,L,2dim)

        # 普通attention  (n,L,dim)
        # rumor_commenttext = self.mh_attention(rumor_commenttext, rumor_commenttext, rumor_commenttext)

        # 谣言检测, 处理原文
        rumor_commenttext = rumor_commenttext.permute(0, 2, 1)
        rumor_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_commenttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_commenttext_feature = torch.cat(rumor_commenttext_conv, dim=1)

        ############################################################
        # 处理post文本
        ############################################################
        # rumor_posttext[0](n,L,768)
        rumor_posttext = self.bert(batch.input_ids1[batch.post_index],
                                   attention_mask=batch.attention_mask1[batch.post_index])
        rumor_posttext = rumor_posttext[0]
        # (n,L,dim)
        rumor_posttext = self.fc(rumor_posttext)
        # 正负attention   (n,L,dim)
        rumor_posttext_pos, rumor_posttext_neg = self.mh_attention(rumor_posttext, rumor_posttext, rumor_posttext)
        rumor_posttext = torch.cat([rumor_posttext_pos, rumor_posttext_neg], dim=2) # (n,L,2dim)

        # 普通attention   (n,L,dim)
        # rumor_posttext = self.mh_attention(rumor_posttext, rumor_posttext, rumor_posttext)

        # 谣言检测, 处理原文
        rumor_posttext = rumor_posttext.permute(0, 2, 1)
        rumor_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_posttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_posttext_feature = torch.cat(rumor_posttext_conv, dim=1)

        # (n, out_channels * kernel_num * 2 + out_feats * 2)
        features = torch.cat([rumor_commenttext_feature, rumor_posttext_feature, rumor_graph_comment, rumor_graph_post],
                             dim=1)

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
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y, y_pred, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []
        total = len(data)
        for i, batch in enumerate(data):
            # if i == total - 1:
            #     print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            # with torch.no_grad():
            #     batch.x = batch.x.cuda(device=self.device)
            #     batch.text = batch.text.cuda(device=self.device)
            #     batch.root_index = batch.root_index.cuda(device=self.device)
            #     batch.edge_index = batch.edge_index.cuda(device=self.device)
            #     batch.y = batch.y.cuda(device=self.device)

            # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
            output = self.forward(batch)
            predicted = torch.max(output, dim=1)[1]
            predicted = predicted.data.cpu().numpy().tolist()
            y_pred += predicted
            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        print("acc:{}    f1:{}".format(res['accuracy'], res['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res

# _GAT
class MTL_GAT(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=30, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0

        #
        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.bert = BertModel.from_pretrained('bert-base-uncased', config=modelConfig)
        bert_outputdim = self.bert.config.hidden_size
        self.fc = nn.Linear(bert_outputdim, 300)

        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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
        # for GCN
        # self.fc1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        # for GAT
        self.fc1 = nn.Linear(out_channels * kernel_num * 2 + out_feats * 5 * 2, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=4)

        # self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        # self.gnn_rumor2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.gnn_rumor1 = GATConv(in_channels=in_feats, out_channels=out_feats, heads=5)
        self.gnn_rumor2 = GATConv(in_channels=out_feats * 5, out_channels=out_feats, heads=5)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)
        init.xavier_normal_(self.fc.weight)

    def forward(self, batch):
        ############################################################
        # 处理图
        ############################################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_comment = rumor_graph[batch.root_index]
        rumor_graph_post = rumor_graph[batch.post_index]

        ############################################################
        # 处理comment文本
        ############################################################

        # rumor_commenttext[0](n,L,768)
        rumor_commenttext = self.bert(batch.input_ids1[batch.root_index],
                                      attention_mask=batch.attention_mask1[batch.root_index])
        rumor_commenttext = rumor_commenttext[0]
        # (n,L,dim)
        rumor_commenttext = self.fc(rumor_commenttext)

        # (n,L,dim)
        rumor_commenttext_pos, rumor_commenttext_neg = self.mh_attention(rumor_commenttext, rumor_commenttext,
                                                                         rumor_commenttext)
        # (n,L,2dim)
        rumor_commenttext = torch.cat([rumor_commenttext_pos, rumor_commenttext_neg], dim=2)

        # 谣言检测, 处理原文
        rumor_commenttext = rumor_commenttext.permute(0, 2, 1)
        rumor_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_commenttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_commenttext_feature = torch.cat(rumor_commenttext_conv, dim=1)

        ############################################################
        # 处理post文本
        ############################################################
        # rumor_posttext[0](n,L,768)
        rumor_posttext = self.bert(batch.input_ids1[batch.post_index],
                                   attention_mask=batch.attention_mask1[batch.post_index])
        rumor_posttext = rumor_posttext[0]
        # (n,L,dim)
        rumor_posttext = self.fc(rumor_posttext)
        # (n,L,dim)
        rumor_posttext_pos, rumor_posttext_neg = self.mh_attention(rumor_posttext, rumor_posttext, rumor_posttext)
        # (n,L,2dim)
        rumor_posttext = torch.cat([rumor_posttext_pos, rumor_posttext_neg], dim=2)

        # 谣言检测, 处理原文
        rumor_posttext = rumor_posttext.permute(0, 2, 1)
        rumor_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_posttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_posttext_feature = torch.cat(rumor_posttext_conv, dim=1)

        # (n, out_channels * kernel_num * 2 + out_feats * 2)
        features = torch.cat([rumor_commenttext_feature, rumor_posttext_feature, rumor_graph_comment, rumor_graph_post],
                             dim=1)

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
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y, y_pred, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []
        total = len(data)
        for i, batch in enumerate(data):
            # if i == total - 1:
            #     print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            # with torch.no_grad():
            #     batch.x = batch.x.cuda(device=self.device)
            #     batch.text = batch.text.cuda(device=self.device)
            #     batch.root_index = batch.root_index.cuda(device=self.device)
            #     batch.edge_index = batch.edge_index.cuda(device=self.device)
            #     batch.y = batch.y.cuda(device=self.device)

            # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
            output = self.forward(batch)
            predicted = torch.max(output, dim=1)[1]
            predicted = predicted.data.cpu().numpy().tolist()
            y_pred += predicted
            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        print("acc:{}    f1:{}".format(res['accuracy'], res['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res


class MTL_NOBERT(torch.nn.Module):
    def __init__(self, config, in_feats=300, out_feats=150, dropout_rate=0.5):
        super(MTL, self).__init__()
        self.config = config
        self.best_acc = 0
        self.patience = 0
        embedding_weights = config['embedding_weights']
        V, D = embedding_weights.shape  # 词典的大小
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        # self.embedding_dim = D
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

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

        self.fc1 = nn.Linear((out_channels * kernel_num + out_feats) * 2, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=4)

        self.gnn_rumor1 = GCNConv(in_channels=in_feats, out_channels=out_feats)
        self.gnn_rumor2 = GCNConv(in_channels=out_feats, out_channels=out_feats)

        self.init_weight()
        print(self)
        self.watch = []

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

    def forward(self, batch):
        ############################################################
        # 处理图
        ############################################################
        rumor_graph = self.gnn_rumor1(batch.x, batch.edge_index)
        rumor_graph = self.gnn_rumor2(rumor_graph, batch.edge_index)
        rumor_graph_comment = rumor_graph[batch.root_index]
        rumor_graph_post = rumor_graph[batch.post_index]

        ############################################################
        # 处理comment文本
        ############################################################
        # 对谁预测就处理谁
        rumor_commenttext = batch.text[batch.root_index]
        rumor_commenttext = self.word_embedding(rumor_commenttext)
        # (n,L,dim)
        rumor_commenttext_pos, rumor_commenttext_neg = self.mh_attention(rumor_commenttext, rumor_commenttext,
                                                                         rumor_commenttext)
        # (n,L,2dim)
        rumor_commenttext = torch.cat([rumor_commenttext_pos, rumor_commenttext_neg], dim=2)

        # 谣言检测, 处理原文
        rumor_commenttext = rumor_commenttext.permute(0, 2, 1)
        rumor_commenttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_commenttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_commenttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_commenttext_feature = torch.cat(rumor_commenttext_conv, dim=1)

        ############################################################
        # 处理post文本
        ############################################################
        rumor_posttext = batch.text[batch.post_index]
        rumor_posttext = self.word_embedding(rumor_posttext)
        # (n,L,dim)
        rumor_posttext_pos, rumor_posttext_neg = self.mh_attention(rumor_posttext, rumor_posttext, rumor_posttext)
        # (n,L,2dim)
        rumor_posttext = torch.cat([rumor_posttext_pos, rumor_posttext_neg], dim=2)

        # 谣言检测, 处理原文
        rumor_posttext = rumor_posttext.permute(0, 2, 1)
        rumor_posttext_conv = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(rumor_posttext))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            rumor_posttext_conv.append(pool)

        # (n, out_channels * kernel_num)
        rumor_posttext_feature = torch.cat(rumor_posttext_conv, dim=1)

        # (n, out_channels * kernel_num * 2 + out_feats * 2)
        features = torch.cat([rumor_commenttext_feature, rumor_posttext_feature, rumor_graph_comment, rumor_graph_post],
                             dim=1)

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
            # print(classification_report(y, y_pred, target_names=self.config['target_names'], digits=5))
            print(classification_report(y, y_pred, digits=5))
            # print("Val　set acc:", acc)
            # print("Best val set acc:", self.best_acc)
            print("save model!!!")
        else:
            print("本轮acc{}小于最优acc{}, 不保存模型".format(acc, self.best_acc))
            self.patience += 1

        return acc, res

    def predict(self, data):
        if torch.cuda.is_available():
            self.cuda()
        # 将模型调整为验证模式, 该模式不启用 BatchNormalization 和 Dropout
        self.eval()
        y_pred = []
        y = []
        total = len(data)
        for i, batch in enumerate(data):
            # if i == total - 1:
            #     print(i)
            y.extend(batch.y.data.cpu().numpy().tolist())
            # 测试时, 不需要计算模型参数的导数, 减少内存开销; 不过这里没有冻结模型参数的导数啊
            # with torch.no_grad():
            #     batch.x = batch.x.cuda(device=self.device)
            #     batch.text = batch.text.cuda(device=self.device)
            #     batch.root_index = batch.root_index.cuda(device=self.device)
            #     batch.edge_index = batch.edge_index.cuda(device=self.device)
            #     batch.y = batch.y.cuda(device=self.device)

            # 把下面三行缩进到with torch.no_grad()内了, 这样logits的requires_grad就是False了, 不过对于减少显存貌似没有什么用
            output = self.forward(batch)
            predicted = torch.max(output, dim=1)[1]
            predicted = predicted.data.cpu().numpy().tolist()
            y_pred += predicted
            # y_pred2 = None
        # res = classification_report(y, y_pred, target_names=self.config['target_names'], digits=5, output_dict=True)
        # res = classification_report(y, y_pred, target_names=target_names, digits=5, output_dict=True)
        res = classification_report(y, y_pred, digits=5, output_dict=True)
        print("acc:{}    f1:{}".format(res['accuracy'], res['macro avg']['f1-score']))
        # res = classification_report(y,
        #                             y_pred,
        #                             labels=[0, 1, 2, 3],
        #                             target_names=['support', 'deny', 'comment', 'query'],
        #                             digits=5,
        #                             output_dict=True)
        return y_pred, y, res


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
