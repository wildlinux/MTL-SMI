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
        # print('index',index)
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


class MTLDataset_pheme_mtl(Dataset):
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
        post_index = data1['graph']['post_index']
        y = data1['label']
        # text1
        text1 = data1['text']
        input_ids1 = data1['tokenizer_encoding']['input_ids']
        attention_mask1 = data1['tokenizer_encoding']['attention_mask']


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


class MTLDataset_fortest(Dataset):
    def __init__(self, length):
        self.length = length

    def __len__(self):
        return self.length

    def __getitem__(self, index):
        embed_dim = 1000
        # graph1
        node_num = np.random.randint(low=1, high=50)
        x = np.random.random(size=(node_num, embed_dim))
        edge_num = np.random.randint(low=node_num, high=3 * node_num)
        row = [np.random.randint(low=0, high=node_num) for i in range(edge_num)]
        col = [np.random.randint(low=0, high=node_num) for i in range(edge_num)]
        edge_index = np.asarray([row, col])
        root_index = 0
        y = np.random.randint(low=0, high=2)
        # text1
        text1_len = 32
        # text1 = np.random.random(size=(node_num, text1_len))
        text1 = np.zeros((node_num, text1_len))

        # graph2
        node_num = np.random.randint(low=1, high=50)
        x2 = np.random.random(size=(node_num, embed_dim))
        edge_num = np.random.randint(low=node_num, high=3 * node_num)
        row = [np.random.randint(low=0, high=node_num) for i in range(edge_num)]
        col = [np.random.randint(low=0, high=node_num) for i in range(edge_num)]

        # (1)变量名不能带index, 否则矩阵中的元素值会根据graph1中的节点数在变, 这是错误的
        # (2)必须通过转置操作改成n*2的形式, 否则无法将多个graph2拼接
        edge_indices2 = np.asarray([row, col]).transpose()

        root_indices = 0
        graph2_node_num = node_num
        graph2_edge_num = len(row)
        y2 = np.random.randint(low=0, high=2)

        # text2
        text2_len = 12
        # text2 = np.random.random(size=(node_num, text2_len))
        text2 = np.ones((node_num, text2_len))
        # 注意是否需要添加方括号
        return Data(x=torch.tensor(x, dtype=torch.float32),
                    text1=torch.LongTensor(text1),
                    root_index=torch.LongTensor([root_index]),
                    edge_index=torch.LongTensor(edge_index),
                    y=torch.LongTensor([y]),
                    x2=torch.tensor(x2, dtype=torch.float32),
                    text2=torch.LongTensor(text2),
                    root_indices=torch.LongTensor([root_indices]),
                    edge_indices2=torch.LongTensor(edge_indices2),
                    y2=torch.LongTensor([y2]),
                    graph2_node_num=torch.LongTensor([graph2_node_num]),
                    graph2_edge_num=torch.LongTensor([graph2_edge_num])
                    )


class GraphDatasetRumor(Dataset):
    def __init__(self, fold_x, treeDic, labels):
        self.mid_list = fold_x
        self.treeDic = treeDic
        self.labels = labels

    def __len__(self):
        return len(self.mid_list)

    def __getitem__(self, index):
        id = self.mid_list[index]
        dic = self.treeDic[id]
        # 获取edge_index
        mids = set()
        list_link_relation = dic[id]
        for link_relation in list_link_relation:
            for fathermid, sonsmid in link_relation.items():
                mids.add(fathermid)
                for sonmid in sonsmid:
                    mids.add(sonmid)
        mids = list(mids)
        map_mid_id = {mid: id for i, mid in enumerate(mids)}
        row = []
        col = []
        for link_relation in list_link_relation:
            for fathermid, sonsmid in link_relation.items():
                fatherid = map_mid_id[fathermid]
                for sonmid in sonsmid:
                    sonid = map_mid_id[sonmid]
                    row.append(fatherid)
                    col.append(sonid)
        edge_index = [row, col]
        y = 1 if self.labels[id] == 'rumor' else 0

        # 获取x, 即各个节点的初始特征

        # return Data(x=torch.tensor(data['x'], dtype=torch.float32),
        #             edge_index=torch.LongTensor(edge_index),
        #             y=torch.LongTensor([y]),
        #             root=torch.LongTensor(data['root']),
        #             rootindex=torch.LongTensor([0]))
