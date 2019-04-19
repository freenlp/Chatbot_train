import os
import torch
import yaml
from torch.utils.data.dataset import Dataset
import numpy as np


def read_vocab(vocab_dir):
    """读取词汇表"""
    with open(vocab_dir, 'r') as fp:
        words = [_.strip() for _ in fp.readlines()]
    word_to_id = dict(zip(words, range(len(words))))
    return words, word_to_id


class TrainData(Dataset):

    words = None
    word_to_id = None

    def __init__(self, train_file, vocab_file, sen_len, train_sc=False):

        if TrainData.word_to_id == None and TrainData.words == None:
            TrainData.words, TrainData.word_to_id = read_vocab(vocab_file)

        self.SOS = TrainData.word_to_id['<SOS>']
        self.EOS = TrainData.word_to_id['<EOS>']
        self.train_file = train_file
        self.train_sc = train_sc
        self.train = []
        if not train_sc:
            with open(train_file) as f:
                # data = yaml.load(f)
                # # two dim array list
                # self.train = data['conversations']
                for item in f:
                    item_split = item.strip().split('\t')
                    if len(item_split) != 2:
                        print(item_split)
                        continue
                    self.train.append(item_split)
        else:
            self.get_data(train_file)
        self.sen_len = sen_len

    def get_data(self, train_file):
        with open(train_file) as f:
            for line in f:
                line_split = line.strip().split()
                label = int(line_split[1])
                with open(line_split[0]) as f_second:
                    data = yaml.load(f_second)
                    content = data['conversations']
                    for item in content:
                        tmp_item = item
                        tmp_item.append(label)
                        self.train.append(tmp_item)

    def crop_pad(self, content):
        if len(content) > self.sen_len:
            content = content[0:self.sen_len]
            content[-1] = self.EOS
            return content, self.sen_len
        else:
            pad_start = len(content)
            tmp_zero = [self.EOS]       # * (self.sen_len - len(content))
            content.extend(tmp_zero)
            return content, pad_start

    def __getitem__(self, index):
        line = self.train[index]
        while len(line) !=2 and len(line) !=3:
            print("line format wrong ")
            print(line)
            index += 1
            line = self.train[index]

        content = line[0]
        label = line[1]
        if self.train_sc:
            class_label = line[2]

        # words to ids
        encoder_data_id = [TrainData.word_to_id[x] for x in content if x in TrainData.word_to_id]
        decoder_data_id = [TrainData.word_to_id[x] for x in label if x in TrainData.word_to_id]
        # add sos
        decoder_data_id = [self.SOS] + decoder_data_id

        label_id = [TrainData.word_to_id[x] for x in label if x in TrainData.word_to_id]

        # add eos pad
        encoder_data_id, pad_start_encoder = self.crop_pad(encoder_data_id)

        decoder_data_id, pad_start_decoder = self.crop_pad(decoder_data_id)
        label_id, pad_start_label = self.crop_pad(label_id)

        encoder_data_id = torch.LongTensor(np.array(encoder_data_id, dtype=np.int64))
        decoder_data_id = torch.LongTensor(np.array(decoder_data_id, dtype=np.int64))
        label = torch.LongTensor(np.array(label_id, dtype=np.int64))

        if self.train_sc:
            class_label = torch.LongTensor([class_label])
            return encoder_data_id, decoder_data_id, label, [pad_start_encoder], class_label
        else:
            return encoder_data_id, decoder_data_id, label, [pad_start_encoder]

    def __len__(self):
        return len(self.train)

    def get_sos_id(self):
        return self.SOS

    def get_eos_id(self):
        return self.EOS

    def get_word_by_id(self, id):
        return str(self.words[id])

    def get_num_by_label(self, label):
        num = 0
        for item in self.train:
            if item[2] in label:
                num += 1
        return num

class PredictionData():

    def __init__(self, vocab_file, sen_len):
        self.words, self.word_to_id = read_vocab(vocab_file)
        self.SOS = self.word_to_id['<SOS>']
        self.EOS = self.word_to_id['<EOS>']
        self.vocab_file = vocab_file
        self.sen_len = sen_len

    def crop_pad(self, content):
        if len(content) > self.sen_len:
            content = content[0:self.sen_len]
            # content[-1] = self.EOS
            return content, self.sen_len
        else:
            pad_start = len(content)
            # prediction is save as train
            tmp_zero = [self.EOS]   # * (self.sen_len - len(content))
            content.extend(tmp_zero)
            return content, pad_start

    def get_ids_by_words(self, words):
        print(words)
        encoder_data_id = [self.word_to_id[x] for x in words if x in self.word_to_id]
        decoder_data_id = [self.SOS]

        encoder_data_id, pad_start = self.crop_pad(encoder_data_id)

        encoder_data_id = torch.LongTensor(np.array(encoder_data_id, dtype=np.int64))
        decoder_data_id = torch.LongTensor(np.array(decoder_data_id, dtype=np.int64))

        return encoder_data_id, decoder_data_id


    def get_sos_id(self):
        return self.SOS

    def get_eos_id(self):
        return self.EOS

    def get_word_by_id(self, id):
        return str(self.words[id])
