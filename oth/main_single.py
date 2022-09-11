from __future__ import division
from __future__ import print_function
from utils import *
import os
import torch.nn as nn
import random
from MTL_single import MTL

# nfeat == max len
nfeat = 120
random.seed(123)

from dataset import MTLDataset
from torch_geometric.data import DataLoader


###################


def load_data():
    pass


if __name__ == "__main__":
    data_path_train = 'dataset/MTL_data_train_weibo.pkl'
    data_path_test = 'dataset/MTL_data_test_weibo.pkl'
    data_train = MTLDataset(path=data_path_train)
    data_train_loader = DataLoader(data_train, batch_size=64, shuffle=False)
    data_test = MTLDataset(path=data_path_test)
    data_test_loader = DataLoader(data_test, batch_size=len(data_test), shuffle=False)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    config = {}
    model_path = 'dataset/weibo_model'
    os.makedirs(model_path, exist_ok=True)
    config['save_path'] = os.path.join(model_path, 'BestModel')
    word_embeddings = np.load('dataset/word_embeddings.npy')
    config['embedding_weights'] = word_embeddings

    model = MTL(config, 300, 300).to(device)
    # attack和defense得用两个分类损失 ? 不是的, 一个就可以; 有权重需求的话可以用两个, 由于数据集分布相对均匀, 所以就不考虑权重问题了
    loss = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=0)
    epoches = 6
    total = len(data_train_loader)
    for epoch in range(epoches):
        model.train()
        for i, batch in enumerate(data_train_loader):
            output2 = model.forward(batch)
            model.zero_grad()
            loss2 = loss(output2, batch.y2)
            lossloss = loss2
            lossloss.backward()
            optimizer.step()

            corrects2 = (torch.max(output2, 1)[1].view(batch.y2.size()).data == batch.y2.data).sum()
            accuracy2 = 100 * corrects2 / len(batch.y2)
            print(
                'Batch[{}/{}] - stance loss: {:.6f}  accuracy: {:.4f}%({}/{})'.format(i + 1, total,
                                                                                      loss2.item(),
                                                                                      accuracy2,
                                                                                      corrects2,
                                                                                      batch.y2.size(0)))
        print("epoch{}在测试集上的效果".format(epoch + 1))
        acc2, res2 = model.evaluate(data_test_loader)
        print('acc2', acc2)

    model.load_state_dict(torch.load(config['save_path']))
    y_pred2, y2, res_stance = model.predict(data_test_loader)

    for k, v in res_stance.items():
        print(k, v)
    print("==============================================================")
    print("立场分类准确率:{:.4f}".format(res_stance['accuracy']))
    print("==============================================================")
