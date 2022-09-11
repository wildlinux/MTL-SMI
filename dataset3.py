import os
import numpy as np
import torch
import json
import pickle
import random
from torch.utils.data import Dataset
from torch_geometric.data import Data


class MTLDataset_semeval_singlestance(Dataset):
    def __init__(self, data: list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.length = len(self.data)
        self.tmp_dict = dict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[index][1]
        # graph1
        # graph中要包含的属性: x, embed_dim?, edge_index, y,
        # text中要包含的属性: text, text_len?
        x = data1['graph']['x']
        edge_index = data1['graph']['edge_index']
        root_index = data1['graph']['root_index']
        post_index = data1['graph']['post_index']
        y = data1['label']
        # text1
        text1 = data1['text']
        input_ids1 = data1['tokenizer_encoding']['input_ids']
        attention_mask1 = data1['tokenizer_encoding']['attention_mask']

        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32).cuda(device=self.device),
                    text=torch.LongTensor(text1).cuda(device=self.device),
                    input_ids1=input_ids1.cuda(device=self.device),
                    attention_mask1=attention_mask1.cuda(device=self.device),
                    root_index=torch.LongTensor([root_index]).cuda(device=self.device),
                    post_index=torch.LongTensor([post_index]).cuda(device=self.device),
                    edge_index=torch.LongTensor(edge_index).cuda(device=self.device),
                    y=torch.LongTensor([y]).cuda(device=self.device))


class MTLDataset_semeval_single(Dataset):
    def __init__(self, data: list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.length = len(self.data)
        self.tmp_dict = dict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[index][1]
        # graph1
        # graph中要包含的属性: x, embed_dim?, edge_index, y,
        # text中要包含的属性: text, text_len?
        x = data1['graph']['x']
        edge_index = data1['graph']['edge_index']
        root_index = data1['graph']['root_index']
        post_index = data1['graph']['post_index']
        y = data1['label']
        # text1
        text1 = data1['text']
        input_ids1 = data1['tokenizer_encoding']['input_ids']
        attention_mask1 = data1['tokenizer_encoding']['attention_mask']

        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32).cuda(device=self.device),
                    text=torch.LongTensor(text1).cuda(device=self.device),
                    input_ids1=input_ids1.cuda(device=self.device),
                    attention_mask1=attention_mask1.cuda(device=self.device),
                    root_index=torch.LongTensor([root_index]).cuda(device=self.device),
                    post_index=torch.LongTensor([post_index]).cuda(device=self.device),
                    edge_index=torch.LongTensor(edge_index).cuda(device=self.device),
                    y=torch.LongTensor([y]).cuda(device=self.device))


class MTLDataset_pheme_single(Dataset):
    def __init__(self, data: list):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.length = len(self.data)
        self.tmp_dict = dict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[index][1]
        # graph1
        # graph中要包含的属性: x, embed_dim?, edge_index, y,
        # text中要包含的属性: text, text_len?
        # print('index',index)
        x = data1['graph']['x']
        edge_index = data1['graph']['edge_index']
        root_index = data1['graph']['root_index']
        y = data1['label']
        # text1
        text1 = data1['text']

        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32),
                    text=torch.LongTensor(text1),
                    root_index=torch.LongTensor([root_index]),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([y]))


class MTLDataset_pheme_mtl(Dataset):
    def __init__(self, data: list):
        # self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # index=0 是谣言检测的数据
        # index=1 是立场分类的数据
        self.data = data
        self.length = len(self.data[0])
        self.tmp_dict = dict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[0][index][1]
        length = len(self.data[1])
        data2 = self.data[1][index % length][1]
        ############################
        # graph1
        ############################
        # graph中要包含的属性: x, embed_dim?, edge_index, y,
        # text中要包含的属性: text, text_len?
        # print('index',index)
        x = data1['graph']['x']
        edge_index = data1['graph']['edge_index']
        root_index = data1['graph']['root_index']
        y = data1['label']
        # text1
        text1 = data1['text']

        ############################
        # graph2
        ############################
        x2 = data2['graph']['x']
        # (1)变量名不能带index, 否则矩阵中的元素值会根据graph1中的节点数在变, 这是错误的
        # (2)必须通过转置操作改成n*2的形式, 否则无法将多个graph2拼接
        edge_indices2 = np.asarray(data2['graph']['edge_index']).transpose()

        root_indices = data2['graph']['root_index']  # 非0
        post_indices = data2['graph']['post_index']  # 非0
        y2 = data2['label']

        # text2
        text2 = data2['text']
        graph2_node_num = data2['graph']['node_num']
        graph2_edge_num = data2['graph']['edge_num']

        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32),
                    text1=torch.LongTensor(text1),
                    root_index=torch.LongTensor([root_index]),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([y]),
                    x2=torch.tensor(x2, dtype=torch.float32),
                    text2=torch.LongTensor(text2),
                    root_indices2=torch.LongTensor([root_indices]),
                    post_indices2=torch.LongTensor([post_indices]),
                    edge_indices2=torch.LongTensor(edge_indices2),
                    y2=torch.LongTensor([y2]),
                    graph2_node_num=torch.LongTensor([graph2_node_num]),
                    graph2_edge_num=torch.LongTensor([graph2_edge_num])
                    )


class MTLDataset_semeval_mtl(Dataset):
    def __init__(self, data: list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        # index=0 是谣言检测的数据
        # index=1 是立场分类的数据
        self.data = data
        self.length = len(self.data[0])
        self.tmp_dict = dict()

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[0][index][1]
        length = len(self.data[1])
        data2 = self.data[1][index % length][1]
        ############################
        # graph1
        ############################
        # graph中要包含的属性: x, embed_dim?, edge_index, y,
        # text中要包含的属性: text, text_len?
        # print('index',index)
        x = data1['graph']['x']
        edge_index = data1['graph']['edge_index']
        root_index = data1['graph']['root_index']
        y = data1['label']
        # text1
        text1 = data1['text']
        input_ids1 = data1['tokenizer_encoding']['input_ids']
        attention_mask1 = data1['tokenizer_encoding']['attention_mask']
        #
        postmid = data1['postmid']
        list_commentmid = data1['list_commentmid']
        ############################
        # graph2
        ############################
        x2 = data2['graph']['x']
        # (1)变量名不能带index, 否则矩阵中的元素值会根据graph1中的节点数在变, 这是错误的
        # (2)必须通过转置操作改成n*2的形式, 否则无法将多个graph2拼接
        edge_indices2 = np.asarray(data2['graph']['edge_index']).transpose()

        root_indices = data2['graph']['root_index']  # 非0
        post_indices = data2['graph']['post_index']  # 非0
        y2 = data2['label']

        # text2
        text2 = data2['text']
        input_ids2 = data2['tokenizer_encoding']['input_ids']
        attention_mask2 = data2['tokenizer_encoding']['attention_mask']

        graph2_node_num = data2['graph']['node_num']
        graph2_edge_num = data2['graph']['edge_num']

        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32).cuda(device=self.device),
                    text1=torch.LongTensor(text1).cuda(device=self.device),
                    input_ids1=input_ids1.cuda(device=self.device),
                    attention_mask1=attention_mask1.cuda(device=self.device),
                    root_index=torch.LongTensor([root_index]).cuda(device=self.device),
                    edge_index=torch.LongTensor(edge_index).cuda(device=self.device),
                    y=torch.LongTensor([y]).cuda(device=self.device),
                    postmid=torch.LongTensor([postmid]).cuda(device=self.device),
                    list_commentmid=torch.LongTensor(list_commentmid).cuda(device=self.device),
                    x2=torch.tensor(x2, dtype=torch.float32).cuda(device=self.device),
                    text2=torch.LongTensor(text2).cuda(device=self.device),
                    input_ids2=input_ids2.cuda(device=self.device),
                    attention_mask2=attention_mask2.cuda(device=self.device),
                    root_indices2=torch.LongTensor([root_indices]).cuda(device=self.device),
                    post_indices2=torch.LongTensor([post_indices]).cuda(device=self.device),
                    edge_indices2=torch.LongTensor(edge_indices2).cuda(device=self.device),
                    y2=torch.LongTensor([y2]).cuda(device=self.device),
                    graph2_node_num=torch.LongTensor([graph2_node_num]).cuda(device=self.device),
                    graph2_edge_num=torch.LongTensor([graph2_edge_num]).cuda(device=self.device)
                    )


class MTLDataset_semeval_mtl2(Dataset):
    def __init__(self, data: list):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.data = data
        self.length = len(self.data)

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        data1 = self.data[0][index][1]
        length = len(self.data[1])
        data2 = self.data[1][index % length][1]



        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32).cuda(device=self.device))

