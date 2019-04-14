#!/usr/bin/python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os
import sys
import time
from datetime import timedelta

import numpy as np
from cnews_loader import read_vocab, read_category, batch_iter, process_file, build_vocab


base_dir = 'text_classification_seq2seq/data/chinese'
train_dir = os.path.join(base_dir, 'ai.yml')

save_dir = 'text_classification_seq2seq/data/preproscess'
vocab_dir = os.path.join(save_dir, 'ai.vocab.txt')


if __name__ == '__main__':
    if not os.path.exists(vocab_dir):  # 如果不存在词汇表，重建
        build_vocab(train_dir, vocab_dir, 5000)
