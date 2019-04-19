# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F
from torch.nn.utils.rnn import pack_padded_sequence

class Seq2seq(nn.Module):
    def __init__(self, batch_size, hidden_size, vocab_size, embedding_length, train_sc):
        super(Seq2seq, self).__init__()

        """
        Arguments
        ---------
        batch_size : Size of the batch which is same as the batch_size of the data returned by the TorchText BucketIterator
        output_size : 10 = (pos, neg)
        hidden_sie : Size of the hidden_state of the LSTM
        vocab_size : Size of the vocabulary containing unique words
        embedding_length : Embeddding dimension of GloVe word embeddings
        
        """
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.vocab_size = vocab_size
        self.embedding_length = embedding_length

        self.word_embeddings = nn.Embedding(vocab_size, embedding_length)  # Initializing the look-up table.
        self.encoder_lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, bidirectional=True)

        self.decoder_lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True, bidirectional=True)
        self.decoder_linear = nn.Linear(hidden_size*2, vocab_size)
        self.train_sc = train_sc

        # used for classification
        self.w2 = nn.Linear(2 * hidden_size + embedding_length, hidden_size)
        self.label = nn.Linear(hidden_size, 2)

    def encoder(self, input_sentence, batch_size=None, pad_start=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        input_data = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        # if self.ignore_pad:
        #     input_data = pack_padded_sequence(input_data, pad_start, batch_first=True)
        if batch_size is None:
            h_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(2, self.batch_size, self.hidden_size).cuda())  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
            c_0 = Variable(torch.zeros(2, batch_size, self.hidden_size))
            if torch.cuda.is_available():
                h_0 = h_0.cuda()
                c_0 = c_0.cuda()

        output, hidden = self.encoder_lstm(input_data, (h_0, c_0))
        if self.train_sc:
            word_embedding = torch.cat((output, input_data), 2)
            y = self.w2(word_embedding) # y.size() = (batch_size, num_sequences, hidden_size)
            y = y.permute(0, 2, 1) # y.size() = (batch_size, hidden_size, num_sequences)
            y = F.max_pool1d(y, y.size()[2]) # y.size() = (batch_size, hidden_size, 1)
            y = y.squeeze(2)
            logits = self.label(y)
            return output, hidden, logits
        else:
            return output, hidden


    def decoder(self, input_sentence, hidden, pad_start=None):

        """
        Parameters
        ----------
        input_sentence: input_sentence of shape = (batch_size, num_sequences)
        batch_size : default = None. Used only for prediction on a single sentence after training (batch_size = 1)

        Returns
        -------
        Output of the linear layer containing logits for positive & negative class which receives its input as the final_hidden_state of the LSTM
        final_output.shape = (batch_size, output_size)

        """

        ''' Here we will map all the indexes present in the input sequence to the corresponding word vector using our pre-trained word_embedddins.'''
        output = self.word_embeddings(
            input_sentence)  # embedded input of shape = (batch_size, num_sequences,  embedding_length)
        output = F.relu(output)
        output, hidden = self.decoder_lstm(output, hidden)
        output = self.decoder_linear(output)

        return output, hidden



