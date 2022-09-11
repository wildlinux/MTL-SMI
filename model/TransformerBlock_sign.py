
import torch
import torch.nn.init as init
import torch.nn.functional as F
import torch.nn as nn



class TransformerBlock(nn.Module):

    def __init__(self, input_size, text_len=14, d_k=16, d_v=16, n_heads=8, is_layer_norm=False, attn_dropout=0.1):
        super(TransformerBlock, self).__init__()
        self.n_heads = n_heads
        self.d_k = d_k if d_k is not None else input_size
        self.d_v = d_v if d_v is not None else input_size

        self.is_layer_norm = is_layer_norm
        if is_layer_norm:
            self.layer_morm = nn.LayerNorm(normalized_shape=input_size)

        self.W_q = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_k = nn.Parameter(torch.Tensor(input_size, n_heads * d_k))
        self.W_v = nn.Parameter(torch.Tensor(input_size, n_heads * d_v))

        # sign attention
        self.W_pos = nn.Parameter(torch.Tensor(text_len, text_len))
        self.W_neg = nn.Parameter(torch.Tensor(text_len, text_len))

        self.W_o = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.W_o2 = nn.Parameter(torch.Tensor(d_v*n_heads, input_size))
        self.linear1 = nn.Linear(input_size, input_size)
        self.linear2 = nn.Linear(input_size, input_size)

        self.dropout = nn.Dropout(attn_dropout)
        self.__init_weights__()
        print(self)

    def __init_weights__(self):
        init.xavier_normal_(self.W_q)
        init.xavier_normal_(self.W_k)
        init.xavier_normal_(self.W_v)
        init.xavier_normal_(self.W_o)
        init.xavier_normal_(self.W_o2)

        # sign attention
        init.xavier_normal_(self.W_pos)
        init.xavier_normal_(self.W_neg)

        init.xavier_normal_(self.linear1.weight)
        init.xavier_normal_(self.linear2.weight)

    def FFN(self, X):
        output = self.linear2(F.relu(self.linear1(X)))
        output = self.dropout(output)
        return output

    def scaled_dot_product_attention(self, Q, K, V, episilon=1e-6):
        '''
        :param Q: (*, max_q_words, n_heads, input_size)
        :param K: (*, max_k_words, n_heads, input_size)
        :param V: (*, max_v_words, n_heads, input_size)
        :param episilon:
        :return:
        '''
        temperature = self.d_k ** 0.5
        Q_K = torch.einsum("bqd,bkd->bqk", Q, K) / (temperature + episilon)

        # 正
        Q_K_pos = torch.einsum("bqk,kk->bqk", Q_K, self.W_pos)
        Q_K_score_pos = F.softmax(Q_K_pos, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score_pos = self.dropout(Q_K_score_pos)
        V_att_pos = Q_K_score_pos.bmm(V)  # (*, max_q_words, input_size)

        # 负
        Q_K_neg = torch.einsum("bqk,kk->bqk", Q_K, self.W_neg)
        Q_K_score_neg = -F.softmax(-Q_K_neg, dim=-1)  # (batch_size, max_q_words, max_k_words)
        Q_K_score_neg = self.dropout(Q_K_score_neg)
        V_att_neg = Q_K_score_neg.bmm(V)  # (*, max_q_words, input_size)

        return V_att_pos, V_att_neg


    def multi_head_attention(self, Q, K, V):
        bsz, q_len, _ = Q.size()
        bsz, k_len, _ = K.size()
        bsz, v_len, _ = V.size()

        Q_ = Q.matmul(self.W_q).view(bsz, q_len, self.n_heads, self.d_k)
        K_ = K.matmul(self.W_k).view(bsz, k_len, self.n_heads, self.d_k)
        V_ = V.matmul(self.W_v).view(bsz, v_len, self.n_heads, self.d_v)

        Q_ = Q_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        K_ = K_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_k)
        V_ = V_.permute(0, 2, 1, 3).contiguous().view(bsz*self.n_heads, q_len, self.d_v)

        V_att_pos, V_att_neg = self.scaled_dot_product_attention(Q_, K_, V_)

        V_att_pos = V_att_pos.view(bsz, self.n_heads, q_len, self.d_v)
        V_att_pos = V_att_pos.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads*self.d_v)
        output_pos = self.dropout(V_att_pos.matmul(self.W_o)) # (batch_size, max_q_words, input_size)

        V_att_neg = V_att_neg.view(bsz, self.n_heads, q_len, self.d_v)
        V_att_neg = V_att_neg.permute(0, 2, 1, 3).contiguous().view(bsz, q_len, self.n_heads * self.d_v)
        output_neg = self.dropout(V_att_neg.matmul(self.W_o2))  # (batch_size, max_q_words, input_size)

        return output_pos, output_neg


    def forward(self, Q, K, V):
        '''
        :param Q: (batch_size, max_q_words, input_size)
        :param K: (batch_size, max_k_words, input_size)
        :param V: (batch_size, max_v_words, input_size)
        :return:  output: (batch_size, max_q_words, input_size)  same size as Q
        '''
        V_att_pos, V_att_neg = self.multi_head_attention(Q, K, V)

        if self.is_layer_norm:
            X_pos = self.layer_morm(Q + V_att_pos)  # (batch_size, max_r_words, embedding_dim)
            output_pos = self.layer_morm(self.FFN(X_pos) + X_pos)

            X_neg = self.layer_morm(Q + V_att_neg)  # (batch_size, max_r_words, embedding_dim)
            output_neg = self.layer_morm(self.FFN(X_neg) + X_neg)
        else:
            X_pos = Q + V_att_pos
            output_pos = self.FFN(X_pos) + X_pos

            X_neg = Q + V_att_neg
            output_neg = self.FFN(X_neg) + X_neg
        return output_pos, output_neg
