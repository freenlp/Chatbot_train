import util.data_load as dl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import numpy as np
from model.lstm import Seq2seq

train_file = 'data/chinese/ai.yml'
# 词汇表
vocab_file = 'data/ai.vocab.txt'

train_file = 'data/train.txt'
vocab_file = 'data/vocab.txt'

class TrainModel:
    def __init__(self, train_file, vocab_file, sentence_len):

        self.train_file = train_file
        self.vocab_file = vocab_file
        self.sentence_len = sentence_len
        self.batch_size = 2
        self.vocab_size = 500
        self.learning_rate = 2e-5
        self.hidden_size = 256
        self.embedding_length = 100
        self.train_sc = True
        self.steps = 0
        # model bitch size must be 1
        self.model = Seq2seq(1, self.hidden_size, self.vocab_size, self.embedding_length, self.train_sc)
        if torch.cuda.is_available():
            self.model.cuda()
        self.optim = torch.optim.Adam(filter(lambda p: p.requires_grad, self.model.parameters()))
        self.loss_fn = F.cross_entropy

        self.train_set = dl.TrainData(train_file, vocab_file, sentence_len, self.train_sc)
        self.train_set_len = len(self.train_set)
        self.train_loader = DataLoader(self.train_set,
                                  batch_size=1,
                                  shuffle=True,
                                  num_workers=2,
                                  drop_last=True
                                  )

        self.train_seq_label = [0]
        self.train_seq_label_num = self.train_set.get_num_by_label(self.train_seq_label) + 1
    def clip_gradient(self, clip_value):

        params = list(filter(lambda p: p.grad is not None, self.model.parameters()))
        for p in params:
            p.grad.data.clamp_(-clip_value, clip_value)

    def get_class_loss(self, loss_fn, pre_class, class_label):
        loss = loss_fn(pre_class, class_label)
        predict = torch.max(pre_class, 1)[1].view(1).data
        acc = (predict == class_label.data).float().sum()
        return loss, acc

    def train_class(self, class_label):
        label = class_label[0].item()
        if label in self.train_seq_label:
            return True
        else:
            False

    def train_seq_class(self, idx, batch):
        encoder_input = batch[0]
        decoder_input = batch[1]
        target = batch[2]
        pad_start = batch[3][0]
        class_label = batch[4]
        if torch.cuda.is_available():
            encoder_input = encoder_input.cuda()
            decoder_input_seq = decoder_input.cuda()
            target = target.cuda()
            class_label = class_label.squeeze(1).cuda()

        output, hidden, pre_class = self.model.encoder(encoder_input, 1, pad_start)
        class_loss, class_acc = self.get_class_loss(self.loss_fn, pre_class, class_label)
        decoder_hidden = hidden
        num_corrects = 0
        loss = 0
        if not self.train_class(class_label):
            return None, 0, class_loss, class_acc

        # decoder input more len than target
        sentence_len = decoder_input_seq.shape[1] - 1
        for di in range(sentence_len):
            decoder_input = decoder_input_seq[:, di].unsqueeze(1).detach()
            decoder_output, decoder_hidden = self.model.decoder(
                decoder_input, decoder_hidden)
            decoder_output = decoder_output.squeeze(1)
            loss += self.loss_fn(decoder_output, target[:, di])
            predict_ids = torch.max(decoder_output, 1)[1].view(1).data
            num_corrects += (predict_ids == target[:, di].data).float().sum()

        # loss = loss / sentence_len
        acc = 100.0 * num_corrects / (1 * sentence_len)
        return loss, acc, class_loss, class_acc

    def train_seq(self, idx, batch):
        encoder_input = batch[0]
        decoder_input = batch[1]
        target = batch[2]
        pad_start = batch[3][0]
        if torch.cuda.is_available():
            encoder_input = encoder_input.cuda()
            decoder_input_seq = decoder_input.cuda()
            target = target.cuda()

        output, hidden = self.model.encoder(encoder_input, 1, pad_start)
        decoder_hidden = hidden
        num_corrects = 0
        loss = 0
        # decoder input more len than target
        sentence_len = decoder_input_seq.shape[1] - 1
        for di in range(sentence_len):
            decoder_input = decoder_input_seq[:, di].unsqueeze(1).detach()
            decoder_output, decoder_hidden = self.model.decoder(
                decoder_input, decoder_hidden)
            decoder_output = decoder_output.squeeze(1)
            loss += self.loss_fn(decoder_output, target[:, di])
            predict_ids = torch.max(decoder_output, 1)[1].view(1).data
            num_corrects += (predict_ids == target[:, di].data).float().sum()

        acc = 100.0 * num_corrects / (1 * sentence_len)
        return loss, acc

    def train_batch(self, idx, steps, batch):
        if steps % self.batch_size == 0:
            self.clip_gradient(1e-1)
            self.optim.step()
            self.optim.zero_grad()
        if self.train_sc:
            loss, acc, class_loss, class_acc = self.train_seq_class(idx, batch)
            # for backward
            if loss is not None:
                total_loss = (loss + class_loss) / self.batch_size
                total_loss.backward()
            else:
                loss = torch.LongTensor([0])
                acc = torch.LongTensor([acc])
                total_loss = class_loss / self.batch_size
                total_loss.backward()
            return loss, acc, class_loss, class_acc
        else:
            loss, acc = self.train_seq(idx, batch)
            # for backward
            total_loss = loss / self.batch_size
            total_loss.backward()
            return loss, acc

    def train_epoch(self, train_iter, epoch):
        total_epoch_loss = 0
        total_epoch_acc = 0
        total_epoch_class_loss = 0
        total_epoch_class_acc = 0
        self.model.train()
        for idx, batch in enumerate(train_iter):
            if self.train_sc:
                loss, acc, class_loss, class_acc = self.train_batch(idx, self.steps, batch)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()
                total_epoch_class_loss += class_loss.item()
                total_epoch_class_acc += class_acc.item()
            else:
                loss, acc = self.train_batch(idx, self.steps, batch)
                total_epoch_loss += loss.item()
                total_epoch_acc += acc.item()
            self.steps += 1

        train_loss, train_acc = total_epoch_loss / self.train_seq_label_num, total_epoch_acc / self.train_seq_label_num
        print('Epoch: ' + str(epoch + 1) + ', Train Loss: ' + str(train_loss) + ', Train Acc: ' + str(train_acc))
        if self.train_sc:
            train_loss, train_acc = total_epoch_class_loss / len(train_iter), total_epoch_class_acc / len(train_iter)
            print('Epoch: ' + str(epoch + 1) + ', Train class Loss: ' + str(train_loss) + ', Train class Acc: ' + str(train_acc))

    def train(self):
        for epoch in range(100):
            self.train_epoch(self.train_loader, epoch)

            if epoch % 20 == 0:
                torch.save(self.model.state_dict(), 'save_model/'+ str(epoch + 1) +'_params.pkl')


train = TrainModel(train_file, vocab_file, 80)
train.train()
