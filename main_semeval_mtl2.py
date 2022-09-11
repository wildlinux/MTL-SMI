from utils import *
import os
import torch.nn as nn
import json
import time
import random
import pickle
import pandas as pd
from sklearn.metrics import classification_report, accuracy_score

from model.MTL_semeval_mtl2 import MTL
from dataset2 import MTLDataset_semeval_mtl
from torch_geometric.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.description = "实验参数配置"
parser.add_argument("-t", "--task", help="任务名称/数据集名称", type=str, default="semeval_mtl2")
parser.add_argument("-g", "--gpu_id", help="在gpu_id上执行run.py", type=str, default="0")
parser.add_argument("-c", "--config_name", help="配置文件的名字", type=str, default="0.json")
parser.add_argument("-T", "--thread_name", help="start.py中的线程名字", type=str, default="Thread-0")
parser.add_argument("-d", "--description", help="实验描述, 英文描述, 不带空格", type=str, default="0225_1")
args = parser.parse_args()

# 加载配置文件
config_path = os.path.join('dataset', args.task, 'all_config_json', args.config_name)
with open(config_path, 'r') as f:
    config = json.load(f)


def make_dir():
    model_suffix = 'mtl_single_semeval'
    res_dir = 'exp_result'
    os.makedirs(res_dir, exist_ok=True)

    res_dir = os.path.join(res_dir, args.task)
    os.makedirs(res_dir, exist_ok=True)

    res_dir = os.path.join(res_dir, args.description)
    os.makedirs(res_dir, exist_ok=True)

    res_dir = config['save_path'] = os.path.join(res_dir, 'best_model_in_each_config')
    os.makedirs(res_dir, exist_ok=True)

    config['save_path'] = os.path.join(res_dir, args.thread_name + '_' + 'config' + args.config_name.split(".")[
        0] + '_best_model_weights_' + model_suffix)
    dir_path = os.path.join('exp_result', args.task, args.description)
    os.makedirs(dir_path, exist_ok=True)


def save_exp_res(res: dict):
    # make_dir()
    res['config'] = args.config_name
    file_name = 'exp_result/{}/{}/{}_{}_{}_result.csv'.format(args.task, args.description,
                                                              args.thread_name, args.gpu_id, args.description)
    today = time.strftime("%Y-%m-%d %H:%M:%S")
    # 自定义排序
    columns = ['date', 'task', 'config', 'min_frequency', 'n_heads', 'self_att_dim', 'self_att_layer_norm',
               'maxlen', 'kernel_sizes', 'nb_filters', 'pooling', 'label_weight',
               'seed', 'epochs', 'batch_size', 'ratio', 'dropout', 'lr', 'use_stopwords', 'which_stopwords',
               'one_more_fc', 'target_names', 'macro avg']
    # columns.extend(config['target_names'])
    columns.extend(['0', '1', '2'])
    columns.append('shuffle_data')
    columns.append('lambda')
    columns.append('accuracy')

    # 读取结果文件, 如果不存在则先创建并保存一次, 记得加入header
    if not os.path.exists(file_name):
        df = pd.DataFrame(columns=columns)
        df.to_csv(path_or_buf=file_name, header=True, index=False)

    df = pd.read_csv(filepath_or_buffer=file_name)
    # 手动规定列的顺序
    df = df[columns]
    i = df.shape[0]
    # 写入时间
    df.loc[i, 'date'] = today
    # 写入实验名/数据集名
    df.loc[i, 'task'] = args.task
    # 写入实验配置
    for k, v in config.items():
        if k in columns:
            df.loc[i, k] = str(v)
    for k, v in res.items():
        if k not in df.columns:
            continue
        df.loc[i, k] = str(v)
    df.to_csv(path_or_buf=file_name, header=True, index=False)


def load_model():
    pass


if __name__ == "__main__":
    # 设置随机数种子
    # setup_seed(config['seed'])
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t1 = time.time()

    config['num_classes'] = 3
    config['target_names'] = ['false', 'true', 'unverified']

    make_dir()
    # 删除已有模型, 防止干扰测试; 一直run时, 如果不删除最优的模型, 就会显示最优的结果, 也就是修改参数时不能立刻看到效果
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    word_embeddings = np.load('dataset/word_embeddings_semeval_mtl2.npy')
    config['embedding_weights'] = word_embeddings

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    # model = MTL(config).to(device)
    model = MTL(config)
    # 冻结参数
    for name, param in model.bert.base_model.named_parameters():
        if 'encoder.layer.11' in name:
            print("{}参加训练".format(name))
            continue
        else:
            param.requires_grad = False

    if torch.cuda.is_available():
        model.cuda()

    batch_size = config['batch_size']
    params = model.parameters()
    optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay=0)

    data1_path_train = 'dataset/MTL_SEMEVAL_RUMOR_TRAIN.pkl'
    with open(data1_path_train, 'rb') as f:
        data1_train = pickle.load(f)
    # 是否打乱数据
    if config['shuffle_data']:
        random.seed(config['seed'])
        random.Random(config['seed']).shuffle(data1_train)

    data2_path_train = 'dataset/MTL_SEMEVAL_STANCE_TRAIN.pkl'
    with open(data2_path_train, 'rb') as f:
        data2_train = pickle.load(f)
    data_train = MTLDataset_semeval_mtl(data=[data1_train, data2_train])
    data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)

    data1_path_test = 'dataset/MTL_SEMEVAL_RUMOR_TEST.pkl'
    with open(data1_path_test, 'rb') as f:
        data1_test = pickle.load(f)
    data2_path_test = 'dataset/MTL_SEMEVAL_STANCE_TEST.pkl'
    with open(data2_path_test, 'rb') as f:
        data2_test = pickle.load(f)
    data_test = MTLDataset_semeval_mtl(data=[data1_test, data2_test])
    data_test_loader = DataLoader(data_test, batch_size=50, shuffle=False)

    # weight = torch.from_numpy(np.array(config['label_weight'])).float().cuda(device=device)
    # loss = nn.CrossEntropyLoss(weight=weight)

    los1 = nn.CrossEntropyLoss()
    los2 = nn.CrossEntropyLoss()

    total = len(data_train_loader)
    for epoch in range(config['epochs']):
        print("\nEpoch ", epoch + 1, "/", config['epochs'])
        model.train()
        total = len(data_train_loader)
        for i, batch in enumerate(data_train_loader):
            optimizer.zero_grad()
            output1, output2 = model.forward(batch)
            loss1 = los1(output1, batch.y)
            loss2 = los2(output2, batch.y2)
            lossloss = loss1 + config['lambda'] * loss2
            lossloss.backward()
            optimizer.step()

            corrects1 = (torch.max(output1, 1)[1].view(batch.y.size()).data == batch.y.data).sum()
            accuracy1 = 100 * corrects1 / len(batch.y)
            corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
            accuracy2 = 100 * corrects2 / len(batch.y2)
            print(
                'Batch[{}/{}] - rumor loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                     loss1.item(),
                                                                                     accuracy1,
                                                                                     corrects1,
                                                                                     batch.y.size(0)))

            print(
                'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                      loss2.item(),
                                                                                      accuracy2,
                                                                                      corrects2,
                                                                                      batch.y2.size(0)))

        print("Epoch{}在测试集上的效果".format(epoch + 1))
        acc, res = model.evaluate(data_test_loader)

    model.load_state_dict(torch.load(config['save_path']))
    y_pred1, y_pred2, y1, y2, res1, res2 = model.predict(data_test_loader)
    for k, v in res1.items():
        print(k, v)
    print("==============================================================")
    print("RUMOR测试结果:")
    print("ACC:{:.4f}".format(res1['accuracy']))
    print("F1:{:.4f}".format(res1['macro avg']['f1-score']))
    print("STANCE测试结果:")
    print("ACC:{:.4f}".format(res2['accuracy']))
    print("F1:{:.4f}".format(res2['macro avg']['f1-score']))
    print("==============================================================")
    save_exp_res(res1)

'''
改
import 相关
config['num_classes'] = 3
config['target_names'] = ['false', 'true', 'unverified']
word_embeddings = np.load('dataset/word_embeddings_semeval_mtl.npy')
'''
