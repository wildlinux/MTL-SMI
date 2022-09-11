import pandas as pd
import os
from pprint import pprint
import argparse
import re
import json

parser = argparse.ArgumentParser()
parser.description = "实验参数配置"
parser.add_argument("-t", "--task", help="任务名称/数据集名称", type=str, default="pheme_mtl_onlytransformer")
parser.add_argument("-d", "--description", help="实验描述, 英文描述, 不带空格", type=str, default="exp_description")
args = parser.parse_args()


def get_cur_result(task: str, description: str):
    dir_name = 'exp_result/{}/{}'.format(task, description)
    result_csvs = [file for file in os.listdir(dir_name) if file.endswith('.csv')]
    dfs = []
    for csv in result_csvs:
        path = os.path.join(dir_name, csv)
        dfs.append(pd.read_csv(filepath_or_buffer=path))
    df = pd.concat(dfs, ignore_index=True)

    dir_data = 'dataset/{}/all_config_json'.format(task)
    total_nums = len(os.listdir(dir_data))
    done_nums = df.shape[0]
    left_nums = total_nums - done_nums
    print('当前已经完成{}次实验, 还差{}次, 共需完成{}次'.format(done_nums, left_nums, total_nums))
    # 按照ACC降序排序
    # res = df.sort_values(by='accuracy', ascending=False)
    df.loc[:, 'f1'] = 0.0
    for index, row in df.iterrows():
        res = row['macro avg']
        # 将单引号替换为双引号，否则json无法loads
        res = re.sub("'", '"', res)
        res = json.loads(res)['f1-score']
        df.loc[index, 'f1'] = res
    # 根据f1降序排序
    res = df.sort_values(by='f1', ascending=False)
    # res.iloc[:20,[-2,-1]]
    # res.iloc[:,[-2,-1]]
    print('=========================================================================')
    print('=========================================================================')
    # 打印最好的结果
    for j in range(len(res.columns)):
        print(df.columns[j], '\t', res.iloc[0, j])
    print('=========================================================================')
    print('=========================================================================')


def te_csv():
    d = {'A': [3, 6, 6, 7, 9], 'B': [2, 5, 8, 0, 0]}
    df = pd.DataFrame(data=d)
    print('排序前:\n', df)
    res = df.sort_values(by='A', ascending=False)
    print('按照A列的值排序:\n', res)
    res = df.sort_values(by=['A', 'B'], ascending=[False, False])
    print('按照A列B列的值排序:\n', res)


def main():
    # get_cur_result(task='weibo2', description='exp_9720')
    # get_cur_result(task='pheme', description='exp_720')
    # get_cur_result(task='pheme', description='exp_without_self_attention_72')
    # get_cur_result(task='fakenewsnet', description='exp_without_self_attention_756')
    get_cur_result(task=args.task, description=args.description)
    # te_csv()


if __name__ == '__main__':
    main()
