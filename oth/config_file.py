############
# python watch_result.py -t semeval2 -d semeval2_1229_2
############
config = {}
config['repeat'] = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1,
                    ]  # 每套实验配置执行2次

config['seed'] = [11615]

config['epochs'] = [6]  #
config['batch_size'] = [64]  # 3
config['label_weight'] = [[1.0, 1.0, 1.0, 1.0]]
config['maxlen'] = [20]  # 3
config['ratio'] = [[70, 10, 20]]  # 2
config['min_frequency'] = [1]
config['use_stopwords'] = [True]
config['which_stopwords'] = [1]
config['one_more_fc'] = [False]
config['n_heads'] = [12]
config['self_att_dim'] = [16]
config['self_att_layer_norm'] = [False]
config['nb_filters'] = [120]
config['kernel_sizes'] = [[4, 5, 6, 7]]
config['pooling'] = ['max']
config['dropout'] = [0.5]  # 3
config['lr'] = [0.001]