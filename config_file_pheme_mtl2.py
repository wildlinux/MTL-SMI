config = {}
config['repeat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    ]  # 每套实验配置执行2次

config['seed'] = [11615]
# config['lambda'] = [0.6, 0.8, 1.0, 1.2]
config['lambda'] = [1.0]
# config['lambda'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2]
config['shuffle_event'] = [False]
config['epochs'] = [6]  #
config['batch_size'] = [64]  # 3
config['label_weight'] = [[1.0, 1.0, 1.0]]
config['maxlen'] = [20]  # 3
config['ratio'] = [[70, 10, 20]]  # 2
config['min_frequency'] = [1]
config['use_stopwords'] = [True]
config['which_stopwords'] = [1]
config['one_more_fc'] = [False]
config['nb_filters'] = [130]
# config['kernel_sizes'] = [[6, 7, 8]]
config['kernel_sizes'] = [[3, 4, 5]]
# config['kernel_sizes'] = [[4, 5, 6, 7],[3, 4, 5]]
config['pooling'] = ['max']
config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16, 18]
config['n_heads'] = [18]
# config['self_att_dim'] = [14, 16]
config['self_att_dim'] = [14]
config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00006, 0.00005, 0.00001, 0.001]
config['lr'] = [0.00005]
# config['lr'] = [0.001]
# config['lr'] = [0.00006, 0.001]


# config = {}
# config['repeat'] = [1, 1
#                     ]  # 每套实验配置执行2次
#
# config['seed'] = [11615]
# # config['lambda'] = [0.6, 0.8]
# # config['lambda'] = [0.4]
# config['lambda'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
# config['shuffle_event'] = [False]
# config['epochs'] = [9]  #
# config['batch_size'] = [64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [20]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125, 130, 135]
# config['kernel_sizes'] = [[4, 5, 6, 7], [4, 5, 6], [3, 4, 5]]
# # config['kernel_sizes'] = [[4, 5, 6]]
# # config['kernel_sizes'] = [[4, 5, 6, 7],[3, 4, 5]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16, 18]
# config['self_att_dim'] = [14, 16]
# config['self_att_layer_norm'] = [False]
# # config['lr'] = [0.00006, 0.00005, 0.00001, 0.001]
# # config['lr'] = [0.00005]
# config['lr'] = [0.001]
# # config['lr'] = [0.00006, 0.001]


# config = {}
# config['repeat'] = [1, 1
#                     ]  # 每套实验配置执行2次
#
# config['seed'] = [11615]
# # config['lambda'] = [0.6, 0.8]
# # config['lambda'] = [0.4]
# config['lambda'] = [1.0]
# config['shuffle_event'] = [False]
# config['epochs'] = [9]  #
# config['batch_size'] = [64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [20]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125, 130, 135]
# config['kernel_sizes'] = [[4, 5, 6, 7], [4, 5, 6], [3, 4, 5]]
# # config['kernel_sizes'] = [[4, 5, 6]]
# # config['kernel_sizes'] = [[4, 5, 6, 7],[3, 4, 5]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16, 18]
# config['self_att_dim'] = [14, 16]
# config['self_att_layer_norm'] = [False]
# # config['lr'] = [0.00006, 0.00005, 0.00001, 0.001]
# config['lr'] = [0.00005]
# # config['lr'] = [0.00006, 0.001]


# ############
# bert_gcn2 or bert_gat2
# ############
# config = {}
# config['repeat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]  # 每套实验配置执行2次
# config['lambda'] = [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4]
# config['seed'] = [11615]
# config['shuffle_event'] = [False]
# config['epochs'] = [9]  #
# config['batch_size'] = [64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [20]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125]
# config['kernel_sizes'] = [[3, 4, 5]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16]
# config['self_att_dim'] = [14]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00005]

# ############
# bert_gcn2 or bert_gat2
# ############
# config = {}
# config['repeat'] = [1, 1]  # 每套实验配置执行2次
#
# config['seed'] = [11615]
#
# config['shuffle_event'] = [False]
# config['epochs'] = [9]  #
# config['batch_size'] = [64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [20]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125, 130, 135]
# config['kernel_sizes'] = [[4, 5, 6, 7], [4, 5, 6], [3, 4, 5]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16, 18]
# config['self_att_dim'] = [14, 16]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00006, 0.00005, 0.00001, 0.001]


# config = {}
# config['repeat'] = [1, 1, 1, 1]  # 每套实验配置执行2次
#
# # 尝试不打乱训练集
# config['seed'] = [1]
#
# config['shuffle_event'] = [False]
# config['epochs'] = [6]  #
# config['batch_size'] = [64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [20]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125, 130]
# config['kernel_sizes'] = [[4, 5, 6, 7], [3, 4, 5, 6, 7]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [16, 18]
# config['self_att_dim'] = [14, 16]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.001]
