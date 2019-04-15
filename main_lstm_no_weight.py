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
sentence_len = 80
batch_size = 2
vocab_size = 400
learning_rate = 2e-5
hidden_size = 256
embedding_length = 100
ignore_pad = True

train_set = dl.TrainData(train_file, vocab_file, sentence_len)
train_loader = DataLoader(train_set,
                          batch_size=batch_size,
                          shuffle=True,
                          num_workers=2,
                          drop_last=True
                          )


def clip_gradient(model, clip_value):
    params = list(filter(lambda p: p.grad is not None, model.parameters()))
    for p in params:
        p.grad.data.clamp_(-clip_value, clip_value)

def sort_by_pad_start(encoder_input, decoder_input, target, pad_start):
    idx_sort = np.argsort(-pad_start)
    pad_start = pad_start[idx_sort]
    target = target.index_select(0, Variable(idx_sort))
    encoder_input = encoder_input.index_select(0, Variable(idx_sort))
    decoder_input = decoder_input.index_select(0, Variable(idx_sort))
    return encoder_input, decoder_input, target, pad_start


def train_model(model, loss_fn, train_iter, epoch):
    total_epoch_loss = 0
    total_epoch_acc = 0
    model.cuda()
    optim = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
    steps = 0
    model.train()
    sos_id = train_set.get_eos_id()
    for idx, batch in enumerate(train_iter):

        encoder_input = batch[0]
        decoder_input = batch[1]
        target = batch[2]
        pad_start = batch[3][0]
        encoder_input, decoder_input, target, pad_start = sort_by_pad_start(encoder_input, decoder_input, target, pad_start)
        if torch.cuda.is_available():
            encoder_input = encoder_input.cuda()
            decoder_input_seq = decoder_input.cuda()
            target = target.cuda()
        optim.zero_grad()
        output, hidden = model.encoder(encoder_input, batch_size, pad_start)
        # not teach
        decoder_hidden = hidden
        num_corrects = 0
        loss = 0
        for di in range(sentence_len):
            decoder_input = decoder_input_seq[:, di].unsqueeze(1).detach()
            decoder_output, decoder_hidden = model.decoder(
                                        decoder_input, decoder_hidden)
            decoder_output = decoder_output.squeeze()
            loss += loss_fn(decoder_output, target[:, di])
            predict_ids = torch.max(decoder_output, 1)[1].view(batch_size).data
            num_corrects += (predict_ids == target[:, di].data).float().sum()

        acc = 100.0 * num_corrects / (batch_size * sentence_len)
        loss.backward()
        clip_gradient(model, 1e-1)
        optim.step()
        steps += 1
        if steps % 100 == 0:
            print("Epoch: " + str(epoch + 1))
        total_epoch_loss += loss.item()
        total_epoch_acc += acc.item()

    return total_epoch_loss / len(train_iter), total_epoch_acc / len(train_iter)


model = Seq2seq(batch_size, hidden_size, vocab_size, embedding_length, ignore_pad)
loss_fn = F.cross_entropy

for epoch in range(100):
    train_loss, train_acc = train_model(model, loss_fn, train_loader, epoch)

    print('Epoch: ' + str(epoch + 1) + ', Train Loss: ' + str(train_loss) + ', Train Acc: ' + str(train_acc))
    if epoch % 20 == 0:
        torch.save(model.state_dict(), 'save_model/'+ str(epoch + 1) +'_params.pkl')
