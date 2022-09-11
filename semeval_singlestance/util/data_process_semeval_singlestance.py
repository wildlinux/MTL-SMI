import json
import os
import re
import random
import time
import pickle
import pandas as pd
import numpy as np
from preprocess import build_vocab_word2vec, clean_str_cut, build_input_data, config, get_embedding_of_sentence, \
get_nodeembedding_matrix
from collections import Counter
from collections import deque
import hashlib
import itertools
import gensim
print(os.getcwd())
import argparse


# w2v_path = 'twitter_w2v_bert_pheme_single.bin'
def get_list_mid_textcutted():
    map_mid_text = {}
    map_trainmid_text = {}
    map_devmid_text = {}
    map_mid_sourcemid = {}
    map_sourcemid_structure = {}
    # tweet mid和用户mid的对应关系
    map_mid_usermid = {}
    map_sourcemid_fathersonmid = {}
    list_train_mid = []
    list_dev_mid = []
    list_test_mid = []
    # subtaskA是立场分类
    path = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
    with open(path, 'r') as f:
        map_trainmid_label = json.load(f)

    path = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
    with open(path, 'r') as f:
        map_devmid_label = json.load(f)

    dir_ = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/rumoureval-data'
    for event in os.listdir(dir_):
        for mid in os.listdir(os.path.join(dir_, event)):
            with open(os.path.join(dir_, event, mid, 'structure.json'), 'r') as f:
                structure = json.load(f)
                map_sourcemid_structure[mid] = structure

            # post可能也有立场标签
            if mid in map_trainmid_label:
                list_train_mid.append(mid)
                map_mid_sourcemid[mid] = mid
                path = os.path.join(dir_, event, mid, 'source-tweet', '{}.json'.format(mid))
                with open(path, 'r') as f:
                    res = json.load(f)
                    map_trainmid_text[mid] = res['text']
                    # 记录该tweet的作者mid
                    usermid = res['user']['id_str']
                    map_mid_usermid[mid] = usermid
                    # 记录作者的描述信息的文本
                    if res['user']['description'] == None:
                        user_info = 'placeholder'
                    else:
                        user_info = res['user']['description']
                    map_mid_text[usermid] = user_info

            elif mid in map_devmid_label:
                list_dev_mid.append(mid)
                map_mid_sourcemid[mid] = mid
                path = os.path.join(dir_, event, mid, 'source-tweet', '{}.json'.format(mid))
                with open(path, 'r') as f:
                    res = json.load(f)
                    map_devmid_text[mid] = res['text']
                    # 记录该tweet的作者mid
                    usermid = res['user']['id_str']
                    map_mid_usermid[mid] = usermid
                    # 记录作者的描述信息的文本
                    if res['user']['description'] == None:
                        user_info = 'placeholder'
                    else:
                        user_info = res['user']['description']
                    map_mid_text[usermid] = user_info

            path = os.path.join(dir_, event, mid, 'replies')
            # 有的post没有评论
            if not os.path.exists(path):
                continue
            for comment in os.listdir(path):
                filepath = os.path.join(path, comment)
                with open(filepath, 'r') as f:
                    res = json.load(f)
                    commentmid = str(res['id'])
                    map_mid_sourcemid[commentmid] = mid
                    if commentmid in map_trainmid_label:
                        list_train_mid.append(commentmid)
                        map_trainmid_text[commentmid] = res['text']
                        # 记录该tweet的作者mid
                        usermid = res['user']['id_str']
                        map_mid_usermid[commentmid] = usermid
                        # 记录作者的描述信息的文本
                        if res['user']['description'] == None:
                            user_info = 'placeholder'
                        else:
                            user_info = res['user']['description']
                        map_mid_text[usermid] = user_info
                    elif commentmid in map_devmid_label:
                        list_dev_mid.append(commentmid)
                        map_devmid_text[commentmid] = res['text']
                        # 记录该tweet的作者mid
                        usermid = res['user']['id_str']
                        map_mid_usermid[commentmid] = usermid
                        # 记录作者的描述信息的文本
                        if res['user']['description'] == None:
                            user_info = 'placeholder'
                        else:
                            user_info = res['user']['description']
                        map_mid_text[usermid] = user_info

    # 测试集标签
    with open('semeval/subtaska.json', 'r') as f:
        map_testmid_label = json.load(f)

    dir_test = 'semeval/rumoureval2017-test/semeval2017-task8-test-data'
    map_testmid_text = {}
    count = 0
    for mid in os.listdir(dir_test):
        with open(os.path.join(dir_test, mid, 'structure.json'), 'r') as f:
            structure = json.load(f)
            map_sourcemid_structure[mid] = structure

        # post可能也有立场标签
        if mid in map_testmid_label:
            map_mid_sourcemid[mid] = mid
            list_test_mid.append(mid)
            path = os.path.join(dir_test, mid, 'source-tweet', '{}.json'.format(mid))
            with open(path, 'r') as f:
                res = json.load(f)
                map_testmid_text[mid] = res['text']
                # 记录该tweet的作者mid
                usermid = res['user']['id_str']
                map_mid_usermid[mid] = usermid
                # 记录作者的描述信息的文本
                if res['user']['description'] == None:
                    user_info = 'placeholder'
                else:
                    user_info = res['user']['description']
                map_mid_text[usermid] = user_info

        path = os.path.join(dir_test, mid, 'replies')
        for comment in os.listdir(path):
            filepath = os.path.join(path, comment)
            with open(filepath, 'r') as f:
                res = json.load(f)
                commentmid = str(res['id'])
                list_test_mid.append(commentmid)
                map_mid_sourcemid[commentmid] = mid
                map_testmid_text[commentmid] = res['text']
                count += 1
                # 记录该tweet的作者mid
                usermid = res['user']['id_str']
                map_mid_usermid[commentmid] = usermid
                # 记录作者的描述信息的文本
                if res['user']['description'] == None:
                    user_info = 'placeholder'
                else:
                    user_info = res['user']['description']
                map_mid_text[usermid] = user_info

    def create_datalist(map_mid_text: dict):
        list_text = []
        list_mid = []
        for mid, text in map_mid_text.items():
            text = re.sub('[\\t\\n\\r]', ' ', text)
            list_text.append((mid, text))
            list_mid.append(mid)
        return list_text, list_mid

    list_train, _ = create_datalist(map_trainmid_text)  # 4238
    list_dev, _ = create_datalist(map_devmid_text)  # 281
    list_test, _ = create_datalist(map_testmid_text)  # 1049

    list_mid_textcutted = []
    # 暂时不考虑list_dev
    for item in list_train + list_test:
        mid = item[0]
        text = item[1]
        text = clean_str_cut(text, 'eng')
        list_mid_textcutted.append((mid, text))

    return list_mid_textcutted


list_mid_textcutted = get_list_mid_textcutted()


# 保存词典，之后可以使用bert对这些词进行初始化
def save_vocabwords():
    sentences = [x[1] for x in list_mid_textcutted]
    # Build vocabulary
    vocabulary_inv = []
    word_counts = Counter(itertools.chain(*sentences))
    # Mapping from index to word; 挑选出出现次数大于等于2的词
    # vocabulary_inv是个列表
    # 按照词的出现次数降序排序
    vocabulary_inv += [x[0] for x in word_counts.most_common() if x[1] >= config['min_frequency']]
    # Mapping from word to index;
    # vocabulary是个字典, key是词, value是词的id; id用来干什么的?
    vocabulary = {x: i for i, x in enumerate(vocabulary_inv)}

    # 导出词典中的词，然后使用bert获取一个新的词典
    list_words = list(vocabulary.keys())
    with open('vocab_word_list_semeval_singlestance.json', 'w') as f:
        json.dump(list_words, f)
    print('词典大小：{}'.format(len(list_words)))


# 是否保存词典
# flag = True
# if flag:
#     save_vocabwords()
    # 接着去执行bert_text_semeval_singlestance.py生成初始词向量,词向量文件名为'twitter_w2v_bert_semeval_singlestance.bin'

##############################################
# 获取Embedding的初始权重和vocabulary
##############################################
sentences = [x[1] for x in list_mid_textcutted]
# 目前有两个词向量矩阵:twitter_w2v_bert.bin和twitter_w2v_bert_semeval_singlestance.bin
# mtl和single用同一个词典就可以
w2v_path = 'twitter_w2v_bert_semeval_singlestance.bin'
if not os.path.exists(w2v_path):
    save_vocabwords()
    python_command = '/home/yangxiaoyu2018/anaconda3/envs/MTGNN/bin/python'
    if not os.path.exists(python_command):
        # Tesla服务器的conda目录
        python_command = '/home/yangxiaoyu2018/.conda/envs/MTGNN/bin/python'
    bert_file_name = 'bert_text_semeval_singlestance.py'
    string = 'cd bert_pretrain; {} {}'.format(python_command, bert_file_name)
    os.system(string)
vocabulary, word_embeddings = build_vocab_word2vec(sentences, w2v_path=w2v_path)
# semeval single 和 semeval mtl用的是同一个词典
np.save('word_embeddings_semeval_singlestance', word_embeddings)
w2v_model = gensim.models.KeyedVectors.load_word2vec_format(fname=w2v_path, binary=True)


# w2v_path = 'twitter_w2v_bert_semeval_singlestance.bin'
def get_semeval_stance_data(vocabulary: dict, w2v_model):
    # 通过bfs将structure转成列表形式
    '''

    :param vocabulary:
    :param w2v_path:
    :return:
    '''

    # 在semeval数据集里, structure是有问题的, structure中的很多mid并没有在semeval中....
    # 最多考虑5个邻居, 从而减少节点数量
    def bfs(map_structure: dict):
        max_neighbor = 5
        queue = deque()
        queue.append(map_structure)
        map_sourcemid_fathersonmid = {}
        while len(queue) != 0:
            # 弹出一个字典
            cur = queue.pop()
            key = list(cur.keys())[0]
            count = 0
            for key2 in cur[key]:
                count += 1
                if count == max_neighbor:
                    break
                if key not in map_sourcemid_fathersonmid:
                    map_sourcemid_fathersonmid[key] = []
                map_sourcemid_fathersonmid[key].append(key2)
                if isinstance(cur[key][key2], dict):
                    queue.append({key2: cur[key][key2]})
        return map_sourcemid_fathersonmid

    map_trainmid_text = {}
    map_devmid_text = {}
    map_mid_sourcemid = {}
    map_sourcemid_structure = {}
    # tweet mid和用户mid的对应关系
    map_mid_usermid = {}
    map_sourcemid_fathersonmid = {}
    list_train_mid = []
    list_dev_mid = []
    list_test_mid = []
    # subtaskA是立场分类
    path = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-train.json'
    with open(path, 'r') as f:
        map_trainmid_label = json.load(f)

    path = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/traindev/rumoureval-subtaskA-dev.json'
    with open(path, 'r') as f:
        map_devmid_label = json.load(f)

    dir_ = 'semeval/semeval2017-task8-dataset/semeval2017-task8-dataset/rumoureval-data'
    for event in os.listdir(dir_):
        for mid in os.listdir(os.path.join(dir_, event)):
            with open(os.path.join(dir_, event, mid, 'structure.json'), 'r') as f:
                structure = json.load(f)
                map_sourcemid_structure[mid] = structure

            # post可能也有立场标签
            if mid in map_trainmid_label:
                list_train_mid.append(mid)
                map_mid_sourcemid[mid] = mid
                path = os.path.join(dir_, event, mid, 'source-tweet', '{}.json'.format(mid))
                with open(path, 'r') as f:
                    res = json.load(f)
                    map_trainmid_text[mid] = res['text']
                    # 记录该tweet的作者mid
                    usermid = res['user']['id_str']
                    map_mid_usermid[mid] = usermid
                    # 记录作者的描述信息的文本
                    if res['user']['description'] == None:
                        user_info = 'placeholder'
                    else:
                        user_info = res['user']['description']
                    map_trainmid_text[usermid] = user_info

            elif mid in map_devmid_label:
                list_dev_mid.append(mid)
                map_mid_sourcemid[mid] = mid
                path = os.path.join(dir_, event, mid, 'source-tweet', '{}.json'.format(mid))
                with open(path, 'r') as f:
                    res = json.load(f)
                    map_devmid_text[mid] = res['text']
                    # 记录该tweet的作者mid
                    usermid = res['user']['id_str']
                    map_mid_usermid[mid] = usermid
                    # 记录作者的描述信息的文本
                    if res['user']['description'] == None:
                        user_info = 'placeholder'
                    else:
                        user_info = res['user']['description']
                    map_devmid_text[usermid] = user_info

            path = os.path.join(dir_, event, mid, 'replies')
            # 有的post没有评论
            if not os.path.exists(path):
                continue
            for comment in os.listdir(path):
                filepath = os.path.join(path, comment)
                with open(filepath, 'r') as f:
                    res = json.load(f)
                    commentmid = str(res['id'])
                    map_mid_sourcemid[commentmid] = mid
                    if commentmid in map_trainmid_label:
                        list_train_mid.append(commentmid)
                        map_trainmid_text[commentmid] = res['text']
                        # 记录该tweet的作者mid
                        usermid = res['user']['id_str']
                        map_mid_usermid[commentmid] = usermid
                        # 记录作者的描述信息的文本
                        if res['user']['description'] == None:
                            user_info = 'placeholder'
                        else:
                            user_info = res['user']['description']
                        map_trainmid_text[usermid] = user_info
                    elif commentmid in map_devmid_label:
                        list_dev_mid.append(commentmid)
                        map_devmid_text[commentmid] = res['text']
                        # 记录该tweet的作者mid
                        usermid = res['user']['id_str']
                        map_mid_usermid[commentmid] = usermid
                        # 记录作者的描述信息的文本
                        if res['user']['description'] == None:
                            user_info = 'placeholder'
                        else:
                            user_info = res['user']['description']
                        map_devmid_text[usermid] = user_info

    # 测试集标签
    with open('semeval/subtaska.json', 'r') as f:
        map_testmid_label = json.load(f)

    dir_test = 'semeval/rumoureval2017-test/semeval2017-task8-test-data'
    map_testmid_text = {}
    count = 0
    for mid in os.listdir(dir_test):
        with open(os.path.join(dir_test, mid, 'structure.json'), 'r') as f:
            structure = json.load(f)
            map_sourcemid_structure[mid] = structure

        # post可能也有立场标签
        if mid in map_testmid_label:
            map_mid_sourcemid[mid] = mid
            list_test_mid.append(mid)
            path = os.path.join(dir_test, mid, 'source-tweet', '{}.json'.format(mid))
            with open(path, 'r') as f:
                res = json.load(f)
                map_testmid_text[mid] = res['text']
                # 记录该tweet的作者mid
                usermid = res['user']['id_str']
                map_mid_usermid[mid] = usermid
                # 记录作者的描述信息的文本
                if res['user']['description'] == None:
                    user_info = 'placeholder'
                else:
                    user_info = res['user']['description']
                map_testmid_text[usermid] = user_info

        path = os.path.join(dir_test, mid, 'replies')
        for comment in os.listdir(path):
            filepath = os.path.join(path, comment)
            with open(filepath, 'r') as f:
                res = json.load(f)
                commentmid = str(res['id'])
                list_test_mid.append(commentmid)
                map_mid_sourcemid[commentmid] = mid
                map_testmid_text[commentmid] = res['text']
                count += 1
                # 记录该tweet的作者mid
                usermid = res['user']['id_str']
                map_mid_usermid[commentmid] = usermid
                # 记录作者的描述信息的文本
                if res['user']['description'] == None:
                    user_info = 'placeholder'
                else:
                    user_info = res['user']['description']
                map_testmid_text[usermid] = user_info

    # 处理structure.json, 为小图做准备
    for event in os.listdir(dir_):
        for mid in os.listdir(os.path.join(dir_, event)):
            with open(os.path.join(dir_, event, mid, 'structure.json'), 'r') as f:
                structure = json.load(f)
                fathersonmid = bfs(structure)
                #####################################
                # 加入用户step1: 找出所有text的mid
                #####################################
                mids = []
                for m, list_mid in fathersonmid.items():
                    if m not in map_mid_usermid:
                        continue
                    mids.append(m)
                    mids.extend(list_mid)
                mids = list(set(mids))

                # mids = []
                # keys_tobedeleted = []
                # for m, list_mid in fathersonmid.items():
                #     if m not in map_mid_usermid:
                #         keys_tobedeleted.append(m)
                #         continue
                #     mids.append(m)
                #     mid_reamined = []
                #     for m2 in list_mid:
                #         if m2 in map_mid_usermid:
                #             mid_reamined.append(m2)
                #             mids.append(m2)
                #     fathersonmid[m] = mid_reamined
                # # 删除没有用到过的节点
                # for key in keys_tobedeleted:
                #     fathersonmid.pop(key)
                # mids = list(set(mids))
                #####################################
                # 加入用户step2: 给每个text添加用户
                #####################################
                usermids = []
                for m in mids:
                    if m not in map_mid_usermid:
                        continue
                    usermid = map_mid_usermid[m]
                    usermids.append(usermid)
                    if m in fathersonmid:
                        fathersonmid[m].append(usermid)
                    else:
                        fathersonmid[m] = [usermid]
                #####################################
                # 加入超级节点  这里是步骤一; 还有步骤二:将超级节点作为root_index
                #####################################
                # fathersonmid['supernode'] = []
                # fathersonmid['supernode'].extend(mids)
                # fathersonmid['supernode'].extend(usermids)
                # 保存当前小图
                map_sourcemid_fathersonmid[mid] = fathersonmid

    # 处理structure.json, 为小图做准备
    dir_test = 'semeval/rumoureval2017-test/semeval2017-task8-test-data'
    for mid in os.listdir(dir_test):
        with open(os.path.join(dir_test, mid, 'structure.json'), 'r') as f:
            structure = json.load(f)
            fathersonmid = bfs(structure)
            #####################################
            # 加入用户step1: 找出所有text的mid
            #####################################
            mids = []
            for m, list_mid in fathersonmid.items():
                if m not in map_mid_usermid:
                    continue
                mids.append(m)
                mids.extend(list_mid)
            mids = list(set(mids))

            # mids = []
            # keys_tobedeleted = []
            # for m, list_mid in fathersonmid.items():
            #     if m not in map_mid_usermid:
            #         keys_tobedeleted.append(m)
            #         continue
            #     mids.append(m)
            #     mid_reamined = []
            #     for m2 in list_mid:
            #         if m2 in map_mid_usermid:
            #             mid_reamined.append(m2)
            #             mids.append(m2)
            #     fathersonmid[m] = mid_reamined
            # # 删除没有用到过的节点
            # for key in keys_tobedeleted:
            #     fathersonmid.pop(key)
            # mids = list(set(mids))
            #####################################
            # 加入用户step2: 给每个text添加用户
            #####################################
            usermids = []
            for m in mids:
                if m not in map_mid_usermid:
                    continue
                usermid = map_mid_usermid[m]
                usermids.append(usermid)
                if m in fathersonmid:
                    fathersonmid[m].append(usermid)
                else:
                    fathersonmid[m] = [usermid]
            #####################################
            # 加入超级节点  这里是步骤一; 还有步骤二:将超级节点作为root_index
            #####################################
            # fathersonmid['supernode'] = []
            # fathersonmid['supernode'].extend(mids)
            # fathersonmid['supernode'].extend(usermids)
            # 保存当前小图
            map_sourcemid_fathersonmid[mid] = fathersonmid

    def create_datalist(map_mid_text: dict):
        list_text = []
        list_mid = []
        for mid, text in map_mid_text.items():
            text = re.sub('[\\t\\n\\r]', ' ', text)
            list_text.append((mid, text))
            list_mid.append(mid)
        return list_text, list_mid

    list_train, _ = create_datalist(map_trainmid_text)
    list_dev, _ = create_datalist(map_devmid_text)
    list_test, _ = create_datalist(map_testmid_text)

    random.seed(config['seed'])
    random.Random(config['seed']).shuffle(list_train)
    random.Random(config['seed']).shuffle(list_dev)
    random.Random(config['seed']).shuffle(list_test)

    random.Random(config['seed']).shuffle(list_train_mid)
    random.Random(config['seed']).shuffle(list_dev_mid)
    random.Random(config['seed']).shuffle(list_test_mid)

    # list_mid_textcutted = []
    map_mid_textcutted = {}
    for item in list_train + list_dev + list_test:
        mid = item[0]
        text = item[1]
        text = clean_str_cut(text, 'eng')
        # list_mid_textcutted.append((mid, text))
        map_mid_textcutted[mid] = text

    # 获取每个小图对应的文本矩阵
    embed_dim = 300

    def get_text_matrix(fathersonmid: dict, curmid: str):
        # 先给节点进行编号
        map_mid_nodeid = {}
        map_nodeid_mid = {}
        id_ = 0
        for fathermid, list_mid in fathersonmid.items():
            if fathermid not in map_mid_nodeid:
                map_nodeid_mid[id_] = fathermid
                map_mid_nodeid[fathermid] = id_
                id_ += 1
            for mid in list_mid:
                map_nodeid_mid[id_] = mid
                map_mid_nodeid[mid] = id_
                id_ += 1

        # 特殊情况：553495625527209985没有对应的小图，原数据集的structure文件有问题
        # 此时直接指定curmid对应索引0即可
        if curmid not in map_mid_nodeid:
            map_nodeid_mid[0] = curmid
            map_mid_nodeid[curmid] = 0
        ####################################################
        # 不使用超级节点
        ####################################################
        root_index = map_mid_nodeid[curmid]

        ####################################################
        # 使用超级节点
        ####################################################
        # root_index = map_mid_nodeid['supernode']

        # 文本 n*length
        text = []
        for id_ in range(len(map_nodeid_mid)):
            mid2 = map_nodeid_mid[id_]
            # if mid2 == '581386094337474560':
            #     print()
            # 有些mid没有文本,让其等于原始post
            if mid2 not in map_mid_textcutted:
                t = map_mid_textcutted[map_nodeid_mid[0]]
            else:
                t = map_mid_textcutted[mid2]
            text.append(t)
        text = build_input_data(text, vocabulary, max_len=config['maxlen'])

        # 初始化节点向量, 得到n*embed_dim的矩阵
        # 现在是进行随机初始化, 之后可以试试其他方法, 比如用文本向量进行初始化
        # 这里要用map_nodeid_mid， 不能用map_mid_nodeid， 因为map_mid_nodeid可能比map_nodeid_mid大1,比如上面的特殊情况
        node_num = len(map_nodeid_mid)
        # x = np.random.random(size=(node_num, embed_dim))
        # 使用文本初始化节点向量
        x = get_nodeembedding_matrix(list_text=text, w2v_model=w2v_model, node_dim=embed_dim)
        # 记录边 edge_index
        row = []
        col = []
        edge_num = 0
        for fathermid, list_mid in fathersonmid.items():
            fatherid = map_mid_nodeid[fathermid]
            for sonmid in list_mid:
                sonid = map_mid_nodeid[sonmid]
                row.append(fatherid)
                col.append(sonid)
                edge_num += 1
        edge_index = [row, col]
        # 这里应该不用转成np array
        # edge_index = np.asarray([row, col])
        return text, x, edge_index, edge_num, node_num, root_index

    # 开始构建小图
    map_mid_label = {}
    map_mid_label.update(map_trainmid_label)
    map_mid_label.update(map_devmid_label)
    map_mid_label.update(map_testmid_label)
    map_label_int = {'support': 0, 'deny': 1, 'comment': 2, 'query': 3}
    data_list_stance_classification = []

    def get_small_graph_and_text(mid: str):
        label = map_mid_label[mid]
        label = map_label_int[label]
        sourcemid = map_mid_sourcemid[mid]
        fathersonmid = map_sourcemid_fathersonmid[sourcemid]
        text, x, edge_index, edge_num, node_num, root_index = get_text_matrix(fathersonmid, mid)
        # 该post对应的数据
        dict_ = {}
        # 标签; 要从0开始
        dict_['label'] = label
        # graph
        # 节点特征向量和边
        dict_['graph'] = {}
        dict_['graph']['x'] = x
        dict_['graph']['edge_index'] = edge_index
        dict_['graph']['edge_num'] = edge_num
        dict_['graph']['node_num'] = node_num
        dict_['graph']['root_index'] = root_index
        dict_['graph']['post_index'] = 0
        # 文本
        dict_['text'] = text
        dict_['root_index'] = root_index
        dict_['commentmid'] = mid
        return (mid, dict_)

    MTL_SEMEVAL_STANCE_TRAIN = []
    MTL_SEMEVAL_STANCE_DEV = []
    MTL_SEMEVAL_STANCE_TEST = []

    for i, mid in enumerate(list_train_mid):
        if i % 1000 == 0:
            print(i, len(list_train_mid))
        mid, dict_ = get_small_graph_and_text(mid)
        MTL_SEMEVAL_STANCE_TRAIN.append((mid, dict_))

    for i, mid in enumerate(list_dev_mid):
        mid, dict_ = get_small_graph_and_text(mid)
        MTL_SEMEVAL_STANCE_DEV.append((mid, dict_))

    for i, mid in enumerate(list_test_mid):
        mid, dict_ = get_small_graph_and_text(mid)
        MTL_SEMEVAL_STANCE_TEST.append((mid, dict_))

    return MTL_SEMEVAL_STANCE_TRAIN, MTL_SEMEVAL_STANCE_DEV, MTL_SEMEVAL_STANCE_TEST

SEMEVAL_SINGLESTANCE_TRAIN, SEMEVAL_SINGLESTANCE_DEV, SEMEVAL_SINGLESTANCE_TEST = get_semeval_stance_data(
    vocabulary=vocabulary, w2v_model=w2v_model)

with open('SEMEVAL_SINGLESTANCE_TRAIN.pkl', 'wb') as f:
    pickle.dump(SEMEVAL_SINGLESTANCE_TRAIN, f)

with open('SEMEVAL_SINGLESTANCE_DEV.pkl', 'wb') as f:
    pickle.dump(SEMEVAL_SINGLESTANCE_DEV, f)

with open('SEMEVAL_SINGLESTANCE_TEST.pkl', 'wb') as f:
    pickle.dump(SEMEVAL_SINGLESTANCE_TEST, f)