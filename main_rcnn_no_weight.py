import os
import time
import util.utils as utils
import data.data_load as dl
import torch
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader
import torch.optim as optim
import numpy as np
from model.rcnn import RCNN
TRAIN_FILE = '/home/longriyao/sdb/test/data/cnews.train.txt'
TEST_FILE = '/home/longriyao/sdb/test/data/cnews.test.txt'
VO_FILE = '/home/longriyao/sdb/test/data/cnews.vocab.txt'

SENTENCE_LEN = 800
BATCH_SIZE = 32
VOCAB_SIZE = 5000

learning_rate = 2e-5
output_size = 10
hidden_size = 256
embedding_length = 100
#TEXT, vocab_size, word_embeddings, train_iter, valid_iter, test_iter = load_data.load_dataset()

train_set = dl.TxtDatasetProcessing(TRAIN_FILE, SENTENCE_LEN)
train_loader = DataLoader(train_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2
                          )

val_set = dl.TxtDatasetProcessing(TEST_FILE, SENTENCE_LEN)
val_loader = DataLoader(val_set,
                          batch_size=BATCH_SIZE,
                          shuffle=True,
                          num_workers=2
                          )

def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)


def sort_by_pad_start(text, target, pad_start):
    idx_sort = np.argsort(-pad_start)
    pad_start = pad_start[idx_sort]
    target = target[idx_sort]
    text = text.index_select(0, Variable(idx_sort))
    return text, target, pad_start


def train_model(model, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    for idx, batch in enumerate(train_iter):
        text = batch[0]
        target = batch[1]
        target = target.view(target.size(0))
        pad_start = batch[2][0]
        # text, target, pad_start = sort_by_pad_start(text, target, pad_start)
        target = torch.autograd.Variable(target).long()
        if torch.cuda.is_available():
            text = text.cuda()
            target = target.cuda()
        if (text.size()[0] is not BATCH_SIZE):  # One of the batch returned by BucketIterator has length different than 32.
            continue
        optim.zero_grad()
        prediction = model(text, BATCH_SIZE)
        # prediction = model(text, BATCH_SIZE, pad_start)
        loss = loss_fn(prediction, target)
        num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).float().sum()
        acc = 100.0 * num_corrects / BATCH_SIZE
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1

        if steps % 100 == 0:
            print('Epoch: ' + str(epoch + 1) + ', Idx: ' + str(idx + 1) + ', Training Loss: ' + str(
                loss.item()) + ', Training Accuracy: ' + str(acc.item()) + ':')
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)

def eval_model(model, val_iter):
    total_epoch_loss = 0
    total_epoch_acc = 0

    total_recall = np.zeros(output_size)
    total_sample = np.zeros(output_size)
    total_right = np.zeros(output_size)

    model.eval()
    with torch.no_grad():
        for idx, batch in enumerate(val_iter):
            text = batch[0]
            target = batch[1]
            target = target.view(target.size(0))
            utils.set_total_sample(target, total_sample)
            pad_start = batch[2][0]
            # text, target, pad_start = sort_by_pad_start(text, target, pad_start)

            if (text.size()[0] is not 32):
                continue
            target = torch.autograd.Variable(target).long()
            if torch.cuda.is_available():
                text = text.cuda()
                target = target.cuda()
            prediction = model(text, BATCH_SIZE)
            # prediction = model(text, BATCH_SIZE, pad_start)
            loss = loss_fn(prediction, target)
            num_corrects = (torch.max(prediction, 1)[1].view(target.size()).data == target.data).sum()
            utils.set_total_right(target, prediction, total_right)
            acc = 100.0 * num_corrects / BATCH_SIZE
            total_epoch_loss += loss.item()
            total_epoch_acc += acc.item()

    total_recall = total_right / total_sample
    return total_epoch_loss / len(val_iter), total_epoch_acc / len(val_iter), total_recall


model = RCNN(BATCH_SIZE, output_size, hidden_size, VOCAB_SIZE, embedding_length)
loss_fn = F.cross_entropy

for epoch in range(100):
    train_loss, train_acc = train_model(model, train_loader, epoch)
    val_loss, val_acc, val_recall = eval_model(model, val_loader)

    print('Epoch: ' + str(epoch + 1) + ', Train Loss: ' + str(train_loss) + ', Train Acc: ' + str(
        train_acc) + ', Val. Loss: ' + str(val_loss) + ', Val. Acc: ' + str(val_acc))
    # 每个类别的recall rate
    for i in range(val_recall.shape[0]):
        print("class " + str(i) + ' recall rate: ' + str(val_recall[i]))

#test_loss, test_acc = eval_model(model, test_iter)
#print('Test Loss: ' + str(test_loss) + ', Test Acc: ' + str(test_acc))

''' Let us now predict the sentiment on a single sentence just for the testing purpose. '''
test_sen1 = "This is one of the best creation of Nolan. I can say, it's his magnum opus. Loved the soundtrack and especially those creative dialogues."
test_sen2 = "Ohh, such a ridiculous movie. Not gonna recommend it to anyone. Complete waste of time and money."

#test_sen1 = TEXT.preprocess(test_sen1)
#test_sen1 = [[TEXT.vocab.stoi[x] for x in test_sen1]]

#test_sen2 = TEXT.preprocess(test_sen2)
#test_sen2 = [[TEXT.vocab.stoi[x] for x in test_sen2]]
"""
test_sen = np.asarray(test_sen1)
test_sen = torch.LongTensor(test_sen)
with torch.no_grad():
    test_tensor = Variable(test_sen)
test_tensor = test_tensor.cuda()
model.eval()
output = model(test_tensor, 1)
out = F.softmax(output, 1)
if (torch.argmax(out[0]) == 1):
    print("Sentiment: Positive")
else:
    print("Sentiment: Negative")
"""
