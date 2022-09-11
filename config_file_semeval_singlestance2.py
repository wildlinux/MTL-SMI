# ############
# bert_gcn2 or bert_gat2
# ############
config = {}
config['repeat'] = [1, 1]  # 每套实验配置执行2次

config['seed'] = [11615]

config['shuffle_data'] = [False]
config['epochs'] = [9]  #
config['batch_size'] = [64]  # 3
config['label_weight'] = [[1.0, 1.0, 1.0]]
config['maxlen'] = [20]  # 3
config['ratio'] = [[70, 10, 20]]  # 2
config['min_frequency'] = [1]
config['use_stopwords'] = [True]
config['which_stopwords'] = [1]
config['one_more_fc'] = [False]
#config['nb_filters'] = [125, 130, 135]
config['nb_filters'] = [ 135]
#config['kernel_sizes'] = [[4, 5, 6, 7], [4, 5, 6], [3, 4, 5]]
config['kernel_sizes'] = [[3, 4, 5]]
config['pooling'] = ['max']
config['dropout'] = [0.5]  # 3
config['n_heads'] = [16, 18]
config['self_att_dim'] = [14, 16]
config['self_att_layer_norm'] = [False]
config['lr'] = [0.00006, 0.00005, 0.00001, 0.001]

# ############
# bert_gcn or bert_gat
# ############
# config = {}
# config['repeat'] = [1, 1, 1]  # 每套实验配置执行2次
#
# config['seed'] = [11615]
#
# config['shuffle_data'] = [False]
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
# config['kernel_sizes'] = [[4, 5, 6, 7], [5, 6, 7], [4, 5, 6], [3, 4, 5]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [14, 16, 18]
# config['self_att_dim'] = [14, 16, 18]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00006]

# ############
# only_bert
# ############
# config = {}
# config['repeat'] = [1] # bert可以稳定复现结果
#
# config['seed'] = [11615]
#
# config['shuffle_data'] = [False]
# config['epochs'] = [9]  #
# config['batch_size'] = [16, 32, 64]  # 3
# config['label_weight'] = [[1.0, 1.0, 1.0]]
# config['maxlen'] = [30]  # 3
# config['ratio'] = [[70, 10, 20]]  # 2
# config['min_frequency'] = [1]
# config['use_stopwords'] = [True]
# config['which_stopwords'] = [1]
# config['one_more_fc'] = [False]
# config['nb_filters'] = [125]
# config['kernel_sizes'] = [[5, 6, 7]]
# config['pooling'] = ['max']
# config['dropout'] = [0.5]  # 3
# config['n_heads'] = [14]
# config['self_att_dim'] = [14]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00004, 0.00006, 0.00005, 0.00003, 0.00002, 0.001]


# ############
# 同一个配置跑多次
# ############
# config = {}
# config['repeat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                     1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
#                     ]  # 每套实验配置执行2次
#
# config['seed'] = [11615]
#
# config['shuffle_data'] = [False]
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
# config['n_heads'] = [18]
# config['self_att_dim'] = [14]
# config['self_att_layer_norm'] = [False]
# config['lr'] = [0.00006]
