# coding: UTF-8
import time
import torch
import numpy as np
from train_eval import train, init_network
from utils import get_time_dif, Preprocess
import argparse

parser = argparse.ArgumentParser(description='Chinese Text Classification')
parser.add_argument('--language', type=str, required=True, help='choose zh or en language~')
args = parser.parse_args()

if __name__ == '__main__':

    #根据 --language 获取运行模型的参数
    if args.language == 'zh':
        from TextCNNzh import Config, Model
        dataset = '../data/THUCNews'
    else:
        from TextCNNen import Config, Model
        dataset = '../data/aclImdb'

    # 设置相同的随机数，保证模型在不同测试的在数据层面的稳定性，同时也会带一些隐性的弊端
    np.random.seed(1)
    torch.manual_seed(1)
    torch.cuda.manual_seed_all(1)
    torch.backends.cudnn.deterministic = True  # 保证每次结果一样

    # 设置对应实验的数据参数、模型参数以及其他设置
    config = Config(dataset)

    start_time = time.time()
    print("Loading data...")
    # 对预训练词向量的embedding进行设置，载入数据分类种类，数据标签和数据样本，并将其文字转化对应的 character level 或者 word level的 idx
    preprocess = Preprocess(config)
    # 获得处理数据集的迭代器，迭代器每次返回一个batch size的 data and labels
    train_iter, dev_iter, test_iter = preprocess.get_iters()

    time_dif = get_time_dif(start_time)
    print("Time usage:", time_dif)

    # train
    # 注意运行环境为 `cuda`
    model = Model(config).to(config.device)
    # 权重初始化
    init_network(model)
    # 打印模型结构
    print(model.parameters)
    # 使用iterators遍历数据喂给训练的模型，同时dev，最后test
    train(config, model, train_iter, dev_iter, test_iter)
