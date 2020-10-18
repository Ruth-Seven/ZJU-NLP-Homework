# coding: UTF-8
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class Config(object):
    """配置参数"""

    def __init__(self, dataset):
        self.model_name = 'TextCNN'
        self.train_path = dataset + '/data/train.txt'                                # 训练集
        self.dev_path = dataset + '/data/dev.txt'                                    # 验证集
        self.test_path = dataset + '/data/test.txt'                                  # 测试集
        self.word_vectors_path = dataset + '/data/word_vectors.pkl'                  # 预训练词向量
        self.class_list = [x.strip() for x in open(
            dataset + '/data/class.txt', encoding='utf-8').readlines()]              # 类别名单
        self.vocab_path = dataset + '/data/vocab.pkl'                                # 单词list
        self.save_path = dataset + '/saved_dict/' + self.model_name + '.ckpt'        # 模型保存地址
        self.log_path = dataset + '/log/' + self.model_name                          # log保存地址
        self.embedding_pretrained = None                                             # 预训练词向量
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')   # 设备
        self.sen_len = 800                                                         # 句子长度
        self.language = 'en'                                                         # 数据集选择
        self.dropout = 0.5                                              # 随机失活
        self.weight_decay = 0.01                                        # 设置weight_decay
        self.set_L2 = True                                             # 设置L2
        self.set_L1 = False                                             # 设置L1
        self.require_improvement = 1000                                 # 若超过1000batch效果还没提升，则提前结束训练
        self.num_classes = len(self.class_list)                         # 类别数
        self.num_epochs = 20                                           # epoch数
        self.batch_size = 374                                           # mini-batch大小
        self.pad_size = 32                                                 # 每句话处理成的长度(短填长切)
        self.learning_rate = 1e-3                                          # 学习率
        self.embed = None                                                    # 字向量维度
        self.filter_sizes = (5,4,3,2)                                   # 卷积核尺寸
        self.num_filters = 64                                         # 卷积核数量(channels数)

    #set embedding
    def set_embedding(self, embedding):
        self.embedding_pretrained = embedding
        self.embed = self.embedding_pretrained.size(1)




class Model(nn.Module):
    def __init__(self, config):
        super(Model, self).__init__()
        # 模型设置embedding
        if config.embedding_pretrained is not None:
            self.embedding = nn.Embedding.from_pretrained(config.embedding_pretrained, freeze=False)
        else:
            self.embedding = nn.Embedding(config.n_vocab, config.embed, padding_idx=config.n_vocab - 1)
        # 设置convolution层的list
        self.convs = nn.ModuleList(
            [nn.Conv2d(1, config.num_filters, (k, config.embed)) for k in config.filter_sizes])
        # 设置失活层
        self.dropout = nn.Dropout(config.dropout)
        # 设置全连接层
        self.fc = nn.Linear(config.num_filters * len(config.filter_sizes), config.num_classes)

    # 计算卷积并池化
    def conv_and_pool(self, x, conv):
        x = F.relu(conv(x)).squeeze(3)
        x = F.max_pool1d(x, x.size(2)).squeeze(2)
        return x

    def forward(self, x):
        out = self.embedding(x)
        out = out.unsqueeze(1)
        out = torch.cat([self.conv_and_pool(out, conv) for conv in self.convs], 1)
        out = self.dropout(out)
        out = self.fc(out)
        # 添加一个softmax
        out = F.softmax(out)
        return out