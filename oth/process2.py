import os
import json
import numpy as np
import random


def process_weibo():
    path = 'dataset/weibo/text_rumor.json'
    with open(path, 'r', encoding='utf-8') as f:
        map_mid_text = json.load(f)
    path = 'dataset/weibo/label_rumor.json'
    with open(path, 'r', encoding='utf-8') as f:
        map_mid_label = json.load(f)  # rumor和non-rumor
    map_mid_id = {mid: i for i, mid in enumerate(map_mid_text.keys())}
    all_mids = list(map_mid_id.keys())
    random.seed(123)
    random.shuffle(all_mids)
    total_num = len(all_mids)
    train_ratio = 0.7
    dev_ratio = 0.1
    train_num = int(total_num * train_ratio)
    dev_num = int(total_num * dev_ratio)
    mid_train = all_mids[:train_num]
    mid_dev = all_mids[train_num:train_num + dev_num]
    mid_test = all_mids[train_num + dev_num:]

    X_train_tid = np.asarray([map_mid_id[mid] for mid in mid_train])
    X_dev_tid = np.asarray([map_mid_id[mid] for mid in mid_dev])
    X_test_tid = np.asarray([map_mid_id[mid] for mid in mid_test])

    X_train = [map_mid_text[mid] for mid in mid_train]
    X_dev = [map_mid_text[mid] for mid in mid_dev]
    X_test = [map_mid_text[mid] for mid in mid_test]

    dic = {'non-rumor': 0, 'rumor': 1}
    y_train = [dic[map_mid_label[mid]] for mid in mid_train]
    y_dev = [dic[map_mid_label[mid]] for mid in mid_dev]
    y_test = [dic[map_mid_label[mid]] for mid in mid_test]
