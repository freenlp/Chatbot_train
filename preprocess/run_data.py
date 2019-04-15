#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import yaml
from collections import Counter
import numpy as np


class DataGen():
    def __init__(self):
        self.all_data = []

    def read_file(self, filename):
        contents, labels = [], []
        with open(filename) as f:
            data = yaml.load(f)
            # two dim array list
            train = data['conversations']
            for item in train:
                content, label = item
                contents.append(content)
                labels.append(label)
        return contents, labels


    def build_vocab(self, train_dir):
        data_train, data_label = self.read_file(train_dir)

        for content in data_train:
            self.all_data.extend(content)

        for content in data_label:
            self.all_data.extend(content)

    def save_vocab(self, vocab_path, vocab_size=None):
        counter = Counter(self.all_data)
        count_pairs = counter.most_common(vocab_size - 1)
        words, _ = list(zip(*count_pairs))
        # 添加一个 <PAD> 来将所有文本pad为同一长度
        words = ['<SOS>'] + list(words)
        words = ['<EOS>'] + list(words)
        with open(vocab_path, "w") as f:
            f.write('\n'.join(words) + '\n')


base_dir = 'data/chinese'
train_dir = os.path.join(base_dir, 'ai.yml')
train_list = []

train_list.append(train_dir)
train_list.append(os.path.join(base_dir, 'movies.yml'))

save_dir = 'data/'
vocab_dir = os.path.join(save_dir, 'vocab.txt')


if __name__ == '__main__':
    data_gen = DataGen()
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        for item in train_list:
            data_gen.build_vocab(item)
        data_gen.save_vocab(vocab_dir, 5000)
