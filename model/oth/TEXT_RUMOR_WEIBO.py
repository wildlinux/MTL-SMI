import torch
import torch.nn as nn
import torch.nn.init as init
from TransformerBlock import TransformerBlock


class TEXT_RUMOR_WEIBO(nn.Module):
    # 传入的adj是coo_matrix数据类型
    def __init__(self, config):
        super(TEXT_RUMOR_WEIBO, self).__init__()
        self.config = config
        # 每一行是一个词的词向量; 大部分都是词典中的词向量(准确), 一小部分词向量是随机生成的
        # embedding_weights = config['embedding_weights']
        # V, D = embedding_weights.shape  # 4336, 300  V是词典中词的个数, D是每个词的维度
        V = 10000
        D = 300
        maxlen = config['maxlen']
        dropout_rate = config['dropout']

        self.mh_attention = TransformerBlock(input_size=300, n_heads=config['n_heads'], attn_dropout=0)
        # self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,
        #                                    _weight=torch.from_numpy(embedding_weights))
        self.word_embedding = nn.Embedding(num_embeddings=V, embedding_dim=D, padding_idx=0,)
        self.convs = nn.ModuleList([nn.Conv1d(300, 100, kernel_size=K) for K in config['kernel_sizes']])
        self.max_poolings = nn.ModuleList([nn.MaxPool1d(kernel_size=maxlen - K + 1) for K in config['kernel_sizes']])

        self.dropout = nn.Dropout(dropout_rate)
        self.relu = nn.ReLU()

        self.fc1 = nn.Linear(300, 300)
        self.fc2 = nn.Linear(in_features=300, out_features=config['num_classes'])

        # self.rnn = nn.RNN(input_size=300, hidden_size=300, num_layers=1, nonlinearity='relu')
        # self.rnn = nn.GRU(input_size=300, hidden_size=300, num_layers=1, bidirectional=False)
        # self.rnn = nn.LSTM(input_size=300, hidden_size=150, num_layers=1, bidirectional=True)
        self.init_weight()
        print(self)

    def init_weight(self):
        init.xavier_normal_(self.fc1.weight)
        init.xavier_normal_(self.fc2.weight)

        # init.xavier_uniform_(self.user_tweet_embedding.weight)

    # 输入的X_text的shape是(64,50), 64是batch size, 50是max_len
    def forward(self, X_text):
        # (N*C, W, D) # X_text在embedding中查询后shape由(64,50)变成了(64,50,300); 更重要的是X_text的requires_grad从False变成了True
        # 这说明X_text中的文本起初都是用词的id表示的, 在embedding中查询后才用词向量表示词, 所以最初的X_text的requires_grad是False
        X_text = self.word_embedding(X_text)
        #########################################################
        # 这个多头attention非常重要, 可以同时在伪装后的adj和原始的adj上保持优秀的结果; 否则不能在原始adj上获取优秀的结果
        X_text = self.mh_attention(X_text, X_text, X_text)
        #########################################################
        # 调整维度的顺序是为了满足pytorch中卷积的参数顺序, (N, input_channels, length)
        X_text = X_text.permute(0, 2, 1)
        ######################################################
        # 卷积对分类效果的提升最大! 所以这里也可以作为超参数进行调整
        conv_block = []
        for _, (Conv, max_pooling) in enumerate(zip(self.convs, self.max_poolings)):
            act = self.relu(Conv(X_text))  # act:128, 100, 48     X_text: 128, 300, 50
            pool = max_pooling(act)  # 128 * 100 * 1
            pool = torch.squeeze(pool)
            conv_block.append(pool)

        conv_feature = torch.cat(conv_block, dim=1)
        features = self.dropout(conv_feature)

        a1 = self.relu(self.fc1(features))
        d1 = self.dropout(a1)
        output = self.fc2(d1)

        return output
