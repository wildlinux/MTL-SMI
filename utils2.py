import random
import torch
from tqdm import tqdm
import json
import os
import numpy as np
from sklearn import metrics
from collections import Counter
import gensim
import numpy as np
import scipy.sparse as sp
import pickle
import itertools
import jieba
import json

jieba.set_dictionary('dict.txt.big')
w2v_dim = 300
config = {'min_frequency': 1, 'maxlen': 50, 'seed': 123}


def get_vocab(path_word_to_index):
    with open(path_word_to_index, 'r', encoding='utf-8') as f:
        w2i = json.load(f)
    return w2i


def get_id_label(path_label):
    map_id_label = {}
    map_label_id = {}
    with open(path_label, 'r', encoding='utf-8') as f:
        for i, label in enumerate(f.readlines()):
            label = label.rstrip()
            map_id_label[i] = label
            map_label_id[label] = i
    return map_id_label, map_label_id


def build_vocab_word2vec(sentences, w2v_path='numberbatch-en.txt'):
    """
    Builds a vocabulary mapping from word to index based on the sentences.
    Returns vocabulary mapping and inverse vocabulary mapping.
    """
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

    print("embedding_weights generation.......")
    # word2vec是个字典; key是词; value是该词的embedding
    word2vec = vocab_to_word2vec(w2v_path, vocabulary)  #
    # 将字典形式的word2vec转成矩阵形式的
    embedding_weights = build_word_embedding_weights(word2vec, vocabulary_inv)
    # vocabulary是个字典, key是词, value是该词对应的id;
    # embedding_weights的第i行是词典中第i个词的向量
    return vocabulary, embedding_weights


# vocab是个字典: key是词, value是词的id
def vocab_to_word2vec(fname, vocab):
    """
    Load word2vec from Mikolov
    """
    np.random.seed(config['seed'])
    word_vecs = {}
    model = gensim.models.KeyedVectors.load_word2vec_format(fname, binary=True)
    count_missing = 0
    for word in vocab:
        if model.__contains__(word):
            word_vecs[word] = model[word]
        else:
            # add unknown words by generating random word vectors
            count_missing += 1
            word_vecs[word] = np.random.uniform(-0.25, 0.25, w2v_dim)

    print(str(len(word_vecs) - count_missing) + " words found in word2vec.")
    print(str(count_missing) + " words not found, generated by random.")
    return word_vecs


# 将字典形式的word2vec转成矩阵形式的
def build_word_embedding_weights(word_vecs, vocabulary_inv):
    """
    Get the word embedding matrix, of size(vocabulary_size, word_vector_size)
    ith row is the embedding of ith word in vocabulary
    """
    vocab_size = len(vocabulary_inv)
    embedding_weights = np.zeros(shape=(vocab_size + 1, w2v_dim), dtype='float32')
    # initialize the first row;
    # 初始化第0行
    # 第0行代表id为0的词, 该词的词向量是全零, 作用是什么? 有的消息长度不足50, 不足的地方用id为0的词表示, 将0补到消息的前面
    embedding_weights[0] = np.zeros(shape=(w2v_dim,))

    # 给embedding_weights的每一行赋值
    # 舍弃了vocabulary_inv[0], 因为vocabulary_inv[0]的位置留给了填充用的词
    for idx in range(1, vocab_size):
        embedding_weights[idx] = word_vecs[vocabulary_inv[idx]]
    print("Embedding matrix of size " + str(np.shape(embedding_weights)))
    return embedding_weights


# X是个list, 一个元素是一条消息, 消息已被分词
def build_input_data(X, vocabulary):
    """
    Maps sentencs and labels to vectors based on a vocabulary.
    """
    # x是个list, 一个元素是一条消息, 消息已被分词, 将每条消息用词的id表示
    x = [[vocabulary[word] for word in sentence if word in vocabulary] for sentence in X]
    # 将每一条消息的长度统一调整为50
    x = pad_sequence(x, max_len=config['maxlen'])
    return x


def pad_sequence(X, max_len=config['maxlen']):
    X_pad = []
    for doc in X:
        if len(doc) >= max_len:
            doc = doc[:max_len]
        else:
            doc = [0] * (max_len - len(doc)) + doc
        X_pad.append(doc)
    return X_pad


def evaluation(model, data_iter):
    model.eval()
    with torch.no_grad():
        acc_list = []
        precision_list = []
        recall_list = []
        f1_list = []
        loss_list = []
        loss = torch.nn.CrossEntropyLoss()
        for btach_x, btach_y in data_iter:
            outputs = model(btach_x)
            p_ = torch.max(outputs.data, 1)[1].cpu().numpy()
            y = btach_y.cpu()
            acc_ = metrics.accuracy_score(y, p_)
            precision_ = metrics.precision_score(y, p_, average='macro')
            recall_ = metrics.recall_score(y, p_, average='macro')
            f1_ = metrics.f1_score(y, p_, average='macro')
            loss_ = loss(outputs, btach_y)
            acc_list.append(acc_)
            precision_list.append(precision_)
            recall_list.append(recall_)
            f1_list.append(f1_)
            loss_list.append(loss_.cpu().data.numpy())

        return np.mean(loss_list), np.mean(acc_list), np.mean(precision_list), np.mean(recall_list), np.mean(f1_list)


def load_dataset(file_path, word_to_index, map_label_id, max_len=32, ngram_size=200000):
    # [PAD]:0    [UNK]:1
    pad_id = word_to_index.get('[PAD]', 0)
    samples = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.read().split("\n")
        lines = [line.strip() for line in lines if len(line) > 0]

    for line in tqdm(lines, desc="load data"):
        line = line.split('\t')
        text = line[0].split(' ')  # 词跟词之间用空格隔开
        label = line[1]
        # [UNK]:1
        text = ([word_to_index.get(word, 1) for word in text]) + [pad_id] * (max_len - len(text))
        text = text[:max_len]
        samples.append(process_data(text, label, map_label_id, max_len, ngram_size))
    return samples


def process_data(text: list, label: str, map_label_id, max_len=32, ngram_size=200000):
    bigram = []
    trigram = []
    id_ = map_label_id[label]
    for i in range(max_len):
        bigram.append(get_bigram_hash(text, i, ngram_size))
        trigram.append(get_trigram_hash(text, i, ngram_size))
    return text, bigram, trigram, id_


def get_bigram_hash(text, index, ngram_size):
    word1 = text[index - 1] if index - 1 >= 0 else 0
    return (word1 * 10600202) % ngram_size


def get_trigram_hash(sequence, index, ngram_size):
    word1 = sequence[index - 1] if index - 1 >= 0 else 0
    word2 = sequence[index - 2] if index - 2 >= 0 else 0
    return (word2 * 10600202 * 13800202 + word1 * 10600202) % ngram_size


class DataIter(object):
    def __init__(self, samples, device, batch_size=32, shuffle=True):
        if shuffle:
            random.shuffle(samples)
        self.samples = samples
        self.batch_size = batch_size
        self.batch_num = len(samples) // self.batch_size
        self.residue = len(samples) % self.batch_num != 0
        self.index = 0
        self.device = device

    def _to_tensor(self, sub_samples: list):
        x = torch.LongTensor([sample[0] for sample in sub_samples]).to(self.device)
        bigram = torch.LongTensor([sample[1] for sample in sub_samples]).to(self.device)
        trigram = torch.LongTensor([sample[2] for sample in sub_samples]).to(self.device)
        y = torch.LongTensor([sample[3] for sample in sub_samples]).to(self.device)
        return (x, bigram, trigram), y

    def __next__(self):
        if self.index == self.batch_num and self.residue:
            sub_samples = self.samples[self.index * self.batch_size: len(self.samples)]
            self.index += 1
            return self._to_tensor(sub_samples)
        elif self.index >= self.batch_num:
            self.index = 0
            raise StopIteration
        else:
            sub_samples = self.samples[self.index * self.batch_size: (self.index + 1) * self.batch_size]
            self.index += 1
            return self._to_tensor(sub_samples)

    def __iter__(self):
        return self

    def __len__(self):
        if self.residue:
            return self.batch_num + 1
        else:
            return self.batch_num
