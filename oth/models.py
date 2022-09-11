import torch.nn as nn
import torch.nn.functional as F
from layers import GraphConvolution
from torch.nn.parameter import Parameter
import torch
import math


class GCN(nn.Module):
    def __init__(self, nfeat, nhid, out, dropout):
        super(GCN, self).__init__()
        self.gc1 = GraphConvolution(nfeat, nhid)
        self.gc2 = GraphConvolution(nhid, out)
        self.dropout = dropout

    def forward(self, x, adj):
        x = F.relu(self.gc1(x, adj))
        x = F.dropout(x, self.dropout, training=self.training)
        x = self.gc2(x, adj)
        return x


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


class SFGCN(nn.Module):
    def __init__(self, nfeat, nclass_rumor=2, nclass_stance=3, nhid1=768, nhid2=256, dropout=0.5):
        super(SFGCN, self).__init__()

        self.SGCN1 = GCN(nfeat, nhid1, nhid2, dropout)
        self.SGCN2 = GCN(nfeat, nhid1, nhid2, dropout)
        self.CGCN = GCN(nfeat, nhid1, nhid2, dropout)

        self.dropout = dropout
        self.a = nn.Parameter(torch.zeros(size=(nhid2, 1)))
        nn.init.xavier_uniform_(self.a.data, gain=1.414)
        # self.attention = Attention(nhid2)
        self.tanh = nn.Tanh()

        # nhid2 * 2:  concatenate emb and com
        self.MLP1 = nn.Sequential(
            nn.Linear(nhid2 * 2, nclass_rumor),
            nn.LogSoftmax(dim=1)
        )

        self.MLP2 = nn.Sequential(
            nn.Linear(nhid2 * 2, nclass_stance),
            nn.LogSoftmax(dim=1)
        )

    def forward(self, x_rumor, x_stance, sadj, fadj):
        emb1 = self.SGCN1(x_rumor, sadj)  # Special_GCN out1 -- sadj structure graph
        emb2 = self.SGCN2(x_stance, fadj)  # Special_GCN out2 -- fadj feature graph
        com1 = self.CGCN(x_rumor, sadj)  # Common_GCN out1 -- sadj structure graph
        com2 = self.CGCN(x_stance, fadj)  # Common_GCN out2 -- fadj feature graph
        # Xcom = (com1 + com2) / 2  # 两个图的节点个数不同, 不能这么操作...

        # emb1 + Xcom  ==> MLP1 ==> output1
        # emb2 + Xcom  ==> MLP2 ==> output2
        # return output1, output2
        input1 = torch.cat((emb1, com1), 1)
        output1 = self.MLP1(input1)

        input2 = torch.cat((emb2, com2), 1)
        output2 = self.MLP2(input2)

        return output1, output2

        ##attention
        # emb = torch.stack([emb1, emb2, Xcom], dim=1)
        # emb, att = self.attention(emb)
        # output = self.MLP(emb)
        # return output, att, emb1, com1, com2, emb2, emb
