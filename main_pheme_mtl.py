from utils import *
import os
import torch.nn as nn
import json
import time
import random
import pickle
import pandas as pd
from model.MTL_pheme_mtl import MTL
from AutomaticWeightedLoss import AutomaticWeightedLoss
from dataset import MTLDataset_pheme_mtl
from torch_geometric.data import DataLoader

import argparse

parser = argparse.ArgumentParser()
parser.description = "实验参数配置"
parser.add_argument("-t", "--task", help="任务名称/数据集名称", type=str, default="pheme_mtl")
parser.add_argument("-g", "--gpu_id", help="在gpu_id上执行run.py", type=str, default="0")
parser.add_argument("-c", "--config_name", help="配置文件的名字", type=str, default="0.json")
parser.add_argument("-T", "--thread_name", help="start.py中的线程名字", type=str, default="Thread-0")
parser.add_argument("-d", "--description", help="实验描述, 英文描述, 不带空格", type=str, default="test1234")
args = parser.parse_args()

# 加载配置文件
config_path = os.path.join('dataset', args.task, 'all_config_json', args.config_name)
with open(config_path, 'r') as f:
    config = json.load(f)


def make_dir():
    model_suffix = 'mtl_pheme'
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
    res['config'] = args.config_name
    # make_dir()
    file_name = 'exp_result/{}/{}/{}_{}_{}_result.csv'.format(args.task, args.description,
                                                              args.thread_name, args.gpu_id, args.description)
    today = time.strftime("%Y-%m-%d %H:%M:%S")
    # 自定义排序
    columns = ['date', 'task', 'config', 'min_frequency', 'n_heads', 'self_att_dim', 'self_att_layer_norm',
               'maxlen', 'kernel_sizes', 'nb_filters', 'pooling', 'label_weight',
               'seed', 'epochs', 'batch_size', 'ratio', 'dropout', 'lr', 'use_stopwords', 'which_stopwords',
               'one_more_fc', 'target_names', 'macro avg']
    columns.extend(config['target_names'])
    columns.append('shuffle_event')
    columns.append('cv_acc')
    columns.append('cv_f1')
    columns.append('avg_acc')
    columns.append('avg_f1')
    columns.append('whole_acc')
    columns.append('whole_f1')
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


# cal(list_pred, list_label)
def cal(list_pred: list, list_label: list):
    t = 0
    length = len(list_pred)
    for i in range(length):
        if list_pred[i] == list_label[i]:
            t += 1
    return t / length


# cal9(list_pred2, list_label2)
def cal9(list_pred2: list, list_label2: list):
    total = len(list_pred2)
    s = 0
    for i in range(total):
        s += cal(list_pred2[i], list_label2[i])
    return s / total


# 计算每个类别的recall, precision, F1
# list_pred和list_label中一共出现几种类别, 就算那几种类别的指标, 没出现的就不考虑了
# calRPF(list_pred, list_label)
def calRPF(list_pred: list, list_label: list, label=0):
    length = len(list_pred)
    TP = TN = FP = FN = 0
    for i in range(length):
        y_pred = list_pred[i]
        y_label = list_label[i]
        if y_pred != label and y_label != label:
            continue
        elif y_pred == label and y_label == label:
            TP += 1
        elif y_pred == label and y_label != label:
            FP += 1
        elif y_pred != label and y_label == label:
            FN += 1
    if TP + FN == 0 or TP + FP == 0:
        return 0.0, 0.0, 0.0
    recall = TP / (TP + FN)
    precision = TP / (TP + FP)
    if recall == 0 and precision == 0:
        return 0.0, 0.0, 0.0
    f1 = 2 * precision * recall / (precision + recall)
    return recall, precision, f1


# calRPFmacro(list_pred, list_label, labels=(0,1,2))
def calRPFmacro(list_pred: list, list_label: list, labels=(0, 1, 2)):
    list_recall = []
    list_precision = []
    list_f1 = []
    for label in labels:
        recall, precision, f1 = calRPF(list_pred, list_label, label)
        list_recall.append(recall)
        list_precision.append(precision)
        list_f1.append(f1)
    # list_pred和list_label中一共出现几种类别, 就算那几种类别的指标, 没出现的就不考虑了
    length = len(set(list_pred + list_label))
    macro_recall = sum(list_recall) / length
    macro_precision = sum(list_precision) / length
    macro_f1 = sum(list_f1) / length
    return macro_recall, macro_precision, macro_f1


# calRPFmacro9(list_pred2, list_label2, labels=(0,1,2))
def calRPFmacro9(list_pred2: list, list_label2: list, labels=(0, 1, 2)):
    list_recall = []
    list_precision = []
    list_f1 = []
    length = len(list_pred2)
    for i in range(length):
        lis_pred2 = list_pred2[i]
        lis_label2 = list_label2[i]
        recall, precision, f1 = calRPFmacro(lis_pred2, lis_label2, (0, 1, 2))
        list_recall.append(recall)
        list_precision.append(precision)
        list_f1.append(f1)
    macro_recall = sum(list_recall) / length
    macro_precision = sum(list_precision) / length
    macro_f1 = sum(list_f1) / length
    # print(list_f1)
    return macro_recall, macro_precision, macro_f1


if __name__ == "__main__":
    # 设置随机数种子
    # setup_seed(config['seed'])
    seed = config['seed']
    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    t1 = time.time()

    config['num_classes'] = 4
    config['target_names'] = ['nonrumor', 'rumor', 'unverified']

    make_dir()
    # 删除已有模型, 防止干扰测试; 一直run时, 如果不删除最优的模型, 就会显示最优的结果, 也就是修改参数时不能立刻看到效果
    if os.path.exists(config['save_path']):
        os.system('rm {}'.format(config['save_path']))

    word_embeddings = np.load('dataset/word_embeddings_pheme_mtl.npy')
    config['embedding_weights'] = word_embeddings

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    data_path = 'dataset/MTL_PHEME_RUMOR_INPUTDATA.pkl'
    with open(data_path, 'rb') as f:
        list_data = pickle.load(f)

    if config['shuffle_event']:
        # 只改变事件的相对顺序, 不改变事件内部的数据顺序
        random.seed(config['seed'])
        random.Random(config['seed']).shuffle(list_data)

    data2_train_path = 'dataset/MTL_SEMEVAL_STANCE_TRAIN.pkl'
    with open(data2_train_path, 'rb') as f:
        list_data2_train = pickle.load(f)

    data2_dev_path = 'dataset/MTL_SEMEVAL_STANCE_DEV.pkl'
    with open(data2_dev_path, 'rb') as f:
        list_data2_dev = pickle.load(f)

    data2_test_path = 'dataset/MTL_SEMEVAL_STANCE_TEST.pkl'
    with open(data2_test_path, 'rb') as f:
        list_data2_test = pickle.load(f)

    # leave one event out
    list_acc = []
    list_f1 = []
    list_test_num = []
    list_train_num = []
    list_traincopy_num = []
    total = []
    list_pred = []
    list_label = []
    list_pred2 = []
    list_label2 = []
    # 9折交叉验证
    for index, data in enumerate(list_data):
        # 每一折都需要重置模型, 否则会沿用上一轮的最优模型, 而上一轮的最优模型已经见过这一轮的测试集了,所以表现会非常好
        model = MTL(config)
        if torch.cuda.is_available():
            model.cuda()
        batch_size = config['batch_size']
        ########################################
        # 不考虑任务的损失权重
        ########################################
        params = model.parameters()
        optimizer = torch.optim.Adam(params, lr=config['lr'], weight_decay=0)

        ########################################
        # 考虑任务的损失权重
        ########################################
        # awl = AutomaticWeightedLoss(2).cuda()
        # optimizer = torch.optim.Adam([
        #     {'params': model.parameters()},
        #     {'params': awl.parameters(), 'weight_decay': 0}
        # ])

        # weight = torch.from_numpy(np.array(config['label_weight'])).float().cuda(device=device)
        # loss = nn.CrossEntropyLoss(weight=weight)
        los1 = nn.CrossEntropyLoss()
        los2 = nn.CrossEntropyLoss()
        # data[index]作为测试集
        data_test_eventname = list_data[index][0]
        data_test1 = list_data[index][1]
        print("==============================================================")
        print(index + 1, data_test_eventname, len(data_test1))
        print("==============================================================")
        data_test2 = list_data2_test
        data_test = MTLDataset_pheme_mtl(data=[data_test1, data_test2])
        data_test_loader = DataLoader(data_test, batch_size=50, shuffle=False)

        data_train1 = []
        for j in range(len(list_data)):
            if j == index:
                continue
            data_train1.extend(list_data[j][1])
        # data_train_copy = data_train.copy()
        # random.seed(config['seed'])
        # random.Random(config['seed']).shuffle(data_train1)
        data_train2 = list_data2_train
        data_train = MTLDataset_pheme_mtl(data=[data_train1, data_train2])
        data_train_loader = DataLoader(data_train, batch_size=batch_size, shuffle=False)
        for epoch in range(config['epochs']):
            # print("\nEpoch ", epoch + 1, "/", config['epochs'])
            model.train()
            total = len(data_train_loader)
            for i, batch in enumerate(data_train_loader):
                # model.zero_grad()#???
                # if i == total - 1:
                #     print(i)
                with torch.no_grad():
                    batch.x = batch.x.cuda(device=device)
                    batch.text1 = batch.text1.cuda(device=device)
                    batch.root_index = batch.root_index.cuda(device=device)
                    batch.edge_index = batch.edge_index.cuda(device=device)
                    batch.y = batch.y.cuda(device=device)

                    batch.x2 = batch.x2.cuda(device=device)
                    batch.text2 = batch.text2.cuda(device=device)
                    batch.root_indices2 = batch.root_indices2.cuda(device=device)
                    batch.post_indices2 = batch.post_indices2.cuda(device=device)
                    batch.edge_indices2 = batch.edge_indices2.cuda(device=device)
                    batch.y2 = batch.y2.cuda(device=device)
                    batch.graph2_node_num = batch.graph2_node_num.cuda(device=device)
                    batch.graph2_edge_num = batch.graph2_edge_num.cuda(device=device)

                optimizer.zero_grad()
                output1, output2 = model.forward(batch)
                loss1 = los1(output1, batch.y)
                loss2 = los2(output2, batch.y2)
                ########################################
                # 不考虑任务的损失权重  15个epoch差不多收敛
                ########################################
                lossloss = loss1 + loss2

                ########################################
                # 考虑任务的损失权重   15个epoch差不多收敛
                ########################################
                # lossloss = awl(loss1, loss2)

                lossloss.backward()
                optimizer.step()

                corrects1 = (torch.max(output1, 1)[1].view(batch.y.size()).data == batch.y.data).sum()
                accuracy1 = 100 * corrects1 / len(batch.y)
                corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
                accuracy2 = 100 * corrects2 / len(batch.y2)
                # print(
                #     'Batch[{}/{}] - rumor loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                #                                                                           loss1.item(),
                #                                                                           accuracy1,
                #                                                                           corrects1,
                #                                                                           batch.y.size(0)))
                #
                # print(
                #     'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                #                                                                          loss2.item(),
                #                                                                          accuracy2,
                #                                                                          corrects2,
                #                                                                          batch.y2.size(0)))

            print("Epoch{}在测试集上的效果".format(epoch + 1))
            acc, res = model.evaluate(data_test_loader)

        model.load_state_dict(torch.load(config['save_path']))
        y_pred1, y_pred2, y1, y2, res1, res2 = model.predict(data_test_loader)
        list_pred2.append(y_pred1)
        list_label2.append(y1)
        list_pred.extend(y_pred1)
        list_label.extend(y1)
        # 将当前的最优结果置0, 否则会影响下一轮实验
        print('将实验最优ACC置0, 避免影响下一组实验')
        model.best_acc = 0
        # 删除已有模型, 防止干扰测试; 一直run时, 如果不删除最优的模型, 就会显示最优的结果, 也就是修改参数时不能立刻看到效果
        if os.path.exists(config['save_path']):
            print('删除当前实验的最优模型')
            os.system('rm {}'.format(config['save_path']))

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
        # save_exp_res(res)
        list_acc.append(res1['accuracy'])
        list_f1.append(res1['macro avg']['f1-score'])
        t2 = time.time()
        # print("共{}个epoch".format(config['epochs']))
        # t = (t2 - t1) / 60
        # print("训练耗时{:.3f}分钟".format(t))
        # print("平均每个epoch耗时{:.3f}分钟".format(t / int(config['epochs'])))
        # break

    res1['cv_acc'] = ' '.join(map(lambda x: '{:.5f}'.format(x), list_acc))
    res1['cv_f1'] = ' '.join(map(lambda x: '{:.5f}'.format(x), list_f1))
    acc_ = sum(list_acc) / 9.0
    res1['accuracy'] = acc_
    f1_ = sum(list_f1) / 9.0
    res1['macro avg']['f1-score'] = f1_
    print('交叉验证后的acc:{}'.format(acc_))
    print('交叉验证后的F1:{}'.format(f1_))
    res1['avg_acc'] = cal9(list_pred2, list_label2)
    res1['avg_f1'] = calRPFmacro9(list_pred2, list_label2, labels=(0, 1, 2))[2]
    res1['whole_acc'] = cal(list_pred, list_label)
    res1['whole_f1'] = calRPFmacro(list_pred, list_label, labels=(0, 1, 2))[2]
    save_exp_res(res1)
    t2 = time.time()
    t = (t2 - t1) / 60
    print("本次训练耗时{:.3f}分钟".format(t))
