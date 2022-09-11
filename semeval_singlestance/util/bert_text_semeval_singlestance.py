from transformers import BertModel, BertConfig, BertTokenizer
import torch
import torch.nn as nn
import numpy as np
import csv
import json
from gensim.models import KeyedVectors
import time
from sklearn.decomposition import PCA

# ——————构造模型——————
class TextNet(nn.Module):
    def __init__(self, code_length):  # code_length为fc映射到的维度大小
        super(TextNet, self).__init__()

        modelConfig = BertConfig.from_pretrained('bert-base-uncased')
        self.textExtractor = BertModel.from_pretrained(
            'bert-base-uncased', config=modelConfig)
        embedding_dim = self.textExtractor.config.hidden_size

        self.fc = nn.Linear(embedding_dim, code_length)

        self.tanh = torch.nn.Tanh()

    def forward(self, tokens, segments, input_masks):
        output = self.textExtractor(tokens, token_type_ids=segments,
                                    attention_mask=input_masks)
        text_embeddings = output[0][:, 0, :]
        # output[0](batch size, sequence length, model hidden dimension)

        # 这里使用PCA会不会更好? 不会
        features = self.fc(text_embeddings)
        # features = self.tanh(features)
        return features


start = time.time()
emb_dim = 300
# emb_dim = 768
textNet = TextNet(code_length=emb_dim)

with open('../vocab_word_list_semeval_singlestance.json', 'r') as f:
    list_words = json.load(f)

# 先用300个词做做实验
# list_words = list_words[:300]
texts = []
for word in list_words:
    raw_text = "[CLS] " + word + " [SEP]"
    texts.append(raw_text)

# ——————输入处理——————
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

# texts = ["[CLS] For those who can't decide between fish or meat.....#Pigfish http: \/\/t.co\/5JBtF54cmg [SEP]",
#          "[CLS] Jim Henson was a puppeteer [SEP]"]

# 将batch size改成1
batch_size = 1
batch_num = int(len(texts) / batch_size)
list_embddings = []
for i in range(batch_num):
    if i != batch_num - 1:
        if i % 200 == 0:
            print(i, batch_num)
        tokens, segments, input_masks = [], [], []
        for text in texts[batch_size * i:batch_size * (i + 1)]:
            tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        # segments列表全0，因为只有一个句子1，没有句子2
        # input_masks列表1的部分代表句子单词，而后面0的部分代表padding，只是用于保持输入整齐，没有实际意义。
        # 相当于告诉BertModel不要利用后面0的部分

        # 转换成PyTorch tensors
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        # ——————提取文本特征——————
        text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
        text_embed = text_hashCodes.detach().numpy()
        # 将shape从(1,300)改成(300,)
        text_embed = text_embed.reshape(emb_dim)
        list_embddings.append(text_embed)
    else:
        tokens, segments, input_masks = [], [], []
        for text in texts[batch_size * i:]:
            tokenized_text = tokenizer.tokenize(text)  # 用tokenizer对句子分词
            indexed_tokens = tokenizer.convert_tokens_to_ids(tokenized_text)  # 索引列表
            tokens.append(indexed_tokens)
            segments.append([0] * len(indexed_tokens))
            input_masks.append([1] * len(indexed_tokens))

        max_len = max([len(single) for single in tokens])  # 最大的句子长度

        for j in range(len(tokens)):
            padding = [0] * (max_len - len(tokens[j]))
            tokens[j] += padding
            segments[j] += padding
            input_masks[j] += padding
        # 转换成PyTorch tensors
        tokens_tensor = torch.tensor(tokens)
        segments_tensors = torch.tensor(segments)
        input_masks_tensors = torch.tensor(input_masks)

        # ——————提取文本特征——————
        text_hashCodes = textNet(tokens_tensor, segments_tensors, input_masks_tensors)  # text_hashCodes是一个32-dim文本特征
        text_embed = text_hashCodes.detach().numpy()
        # 将shape从(1,300)改成(300,)
        text_embed = text_embed.reshape(emb_dim)
        list_embddings.append(text_embed)
#
# 使用PCA从768维降到300维
# pca = PCA(n_components=300)
# low_embeddings = pca.fit_transform(list_embddings)
#
# kv = KeyedVectors(vector_size=300)
# kv.add(list_words, low_embeddings)

kv = KeyedVectors(vector_size=emb_dim)
kv.add(list_words, list_embddings)  # adds those keys (words) & vectors as batch


kv.save_word2vec_format('../twitter_w2v_bert_semeval_singlestance.bin', binary=True)

end = time.time()
total_time = (end - start) / 60
print("耗时:{:.2f}分钟".format(total_time))

'''
改为
with open('../vocab_word_list_semeval_singlestance.json', 'r') as f:
kv.save_word2vec_format('../twitter_w2v_bert_semeval_singlestance.bin', binary=True)


词典大小10227
耗时:13.57分钟
耗时:5.36分钟
'''