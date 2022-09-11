import json, os, time
import threading
import argparse
import config_file_pheme_mtl2

parser = argparse.ArgumentParser()
parser.description = "实验参数配置"
parser.add_argument("-t", "--task", help="任务名称/数据集名称", type=str, default="pheme_mtl2")
parser.add_argument("-g", "--gpu_avail", help="可用的gpu编号,逗号隔开, 如-g 0,1", type=str, default="1")
parser.add_argument("-d", "--description", help="实验描述, 英文描述, 不带空格", type=str, default="0225")
parser.add_argument("-p", "--per", help="每块显卡上跑几个程序", type=int, default="1")
args = parser.parse_args()

script_name = 'main_pheme_mtl2.py'

# 用于列举所有的参数组合可能
def create_all_config_combination(origin_config: dict, task: str) -> list:
    config_dir = os.path.join('dataset', task)
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    config_dir = os.path.join('dataset', task, 'all_config_json')
    if not os.path.exists(config_dir):
        os.mkdir(config_dir)
    n = len(origin_config)
    for k, v in origin_config.items():
        for i, e in enumerate(v):
            v[i] = {k: e}
    res = []
    tmp = {}
    config = [v for v in origin_config.values()]
    dfs(res, tmp, config, 0)
    print('一共有{}种实验配置'.format(len(res)))
    config_dir = 'dataset/{}/all_config_json'.format(task)
    if os.path.exists(config_dir):
        # 删除旧的配置文件们
        os.system('rm -rf {}'.format(config_dir))
        os.mkdir(config_dir)
    for i, config in enumerate(res):
        path = os.path.join(config_dir, str(i) + '.json')
        with open(path, 'w', encoding='utf-8') as f:
            json.dump(config, f)
    # 返回值可以用来查看组合
    return res


def dfs(res=[], tmp={}, config=[], i=0):
    if i == len(config):
        res.append(tmp.copy())
    else:
        for e in config[i]:
            # change spot
            tmp.update(e)
            # new condition,new recursion
            dfs(res, tmp, config, i + 1)
            # restore spot
            tmp.pop(list(e.keys())[0])


def execute(task: str, config_file_names: list, gpu_id: int):
    done_job = 'exp_result'
    os.makedirs(done_job, exist_ok=True)

    done_job = os.path.join(done_job, args.task)
    os.makedirs(done_job, exist_ok=True)

    done_job = os.path.join(done_job, args.description)
    os.makedirs(done_job, exist_ok=True)

    done_job = os.path.join(done_job, 'done_job.txt')
    # 已经运行过的实验配置
    done = []
    if os.path.exists(done_job):
        with open(done_job, 'r') as f:
            for line in f.readlines():
                done.append(line.strip())
    thread_name = threading.current_thread().getName()
    n = len(config_file_names)
    python_command = '/home/yangxiaoyu2018/anaconda3/envs/MTGNN/bin/python'
    if not os.path.exists(python_command):
        # Tesla服务器的conda目录
        python_command = '/home/yangxiaoyu2018/.conda/envs/MTGNN/bin/python'
    for i, config_name in enumerate(config_file_names):
        if config_name in done:
            print('{},该配置已执行过, 跳过'.format(config_name))
            continue
        start = time.time()
        # 执行main_single_semeval.py
        os.system(
            'CUDA_VISIBLE_DEVICES={} {} {} -g {} -t {} -c {} -T {} -d {}'.format(gpu_id,
                                                                                 python_command,
                                                                                 script_name,
                                                                                 gpu_id, task,
                                                                                 config_name,
                                                                                 thread_name,
                                                                                 args.description))
        end = time.time()
        total = '{:.2f}'.format((end - start) / 60)
        today = time.strftime("%Y-%m-%d %H:%M:%S")
        tip = '{} '.format(today) + threading.current_thread().getName() + ' {}/{} 耗时:{}\n'.format(i, n, total)
        print(tip)
        with open('exp_record.txt', 'a', encoding='utf-8') as f:
            f.write(tip)
        with open(done_job, 'a') as f:
            f.write(config_name + '\n')


def multitask2(task: str, gpu_avail: list, per: int):
    config_dir = os.path.join('dataset', task, 'all_config_json')
    all_config_file = os.listdir(config_dir)
    config_file_num = len(all_config_file)
    gpu_num = len(gpu_avail)
    thread_num = per * gpu_num
    # 每个线程处理多少个任务
    avg = int(config_file_num / thread_num)
    config_for_each_thread = []

    map_thread_configfile = {}
    for i in range(config_file_num):
        thread_index = i % thread_num
        if thread_index not in map_thread_configfile:
            map_thread_configfile[thread_index] = []
        map_thread_configfile[thread_index].append(all_config_file[i])

    map_gpu_tasknum = {}
    for i in range(thread_num):
        # 服务器上5块GPU,所以i%5, 根据服务器配置修改这个变量
        print('Thread:{}执行{}次任务'.format(i, len(map_thread_configfile[i])))
        gpu_index = i % gpu_num
        if gpu_index not in map_gpu_tasknum:
            map_gpu_tasknum[gpu_index] = 0
        map_gpu_tasknum[gpu_index] += len(map_thread_configfile[i])
        t = threading.Thread(target=execute, args=(task, map_thread_configfile[i], gpu_avail[gpu_index]))
        t.start()
    print('每块GPU执行的任务数量:', map_gpu_tasknum)


def main():
    print()
    config = config_file_pheme_mtl2.config
    create_all_config_combination(origin_config=config, task=args.task)
    # per表示一个显卡上跑几个程序
    multitask2(task=args.task, gpu_avail=args.gpu_avail.split(","), per=args.per)


if __name__ == '__main__':
    main()


'''
改
import config_file_semeval_singlestance2
config = config_file_semeval_singlestance2.config
parser.add_argument("-t", "--task", help="任务名称/数据集名称", type=str, default="semeval_singlestance2")
os.system()中的.py文件
'''