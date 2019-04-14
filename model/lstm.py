# _*_ coding: utf-8 _*_

import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn import functional as F

class Seq2seq(nn.Module):
    def __init__(self, batch_size, hidden_size, vocab_size, embedding_length):
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
        self.encoder_lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)

        self.decoder_lstm = nn.LSTM(embedding_length, hidden_size, batch_first=True)
        self.decoder_linear = nn.Linear(hidden_size, vocab_size)
        self.decoder_softmax = nn.LogSoftmax(dim=1)

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
        if batch_size is None:
            h_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())  # Initial hidden state of the LSTM
            c_0 = Variable(torch.zeros(1, self.batch_size, self.hidden_size).cuda())  # Initial cell state of the LSTM
        else:
            h_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())
            c_0 = Variable(torch.zeros(1, batch_size, self.hidden_size).cuda())

        output, hidden = self.encoder_lstm(input_data, (h_0, c_0))

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



