import util.data_load as dl
import torch
import numpy as np
from model.lstm import Seq2seq

# 词汇表
vocab_file = 'data/ai.vocab.txt'

sentence_len = 80
vocab_size = 400
batch_size = 1


hidden_size = 256
embedding_length = 100
data_layer = dl.PredictionData(vocab_file, sentence_len)


def prediction(model, question, data_layer):
    if torch.cuda.is_available():
        model.cuda()
    model.eval()

    sos_id = data_layer.get_sos_id()
    eos_id = data_layer.get_eos_id()
    encoder_input, decoder_input = data_layer.get_ids_by_words(question)
    encoder_input = encoder_input.unsqueeze(0)
    if torch.cuda.is_available():
        encoder_input = encoder_input.cuda()
        decoder_input = decoder_input.cuda()

    output, hidden = model.encoder(encoder_input, batch_size, None)
    decoder_hidden = hidden
    decoder_input = decoder_input.unsqueeze(0)
    answer = ""
    for di in range(sentence_len):

        decoder_output, decoder_hidden = model.decoder(
            decoder_input, decoder_hidden)
        id = torch.argmax(decoder_output.squeeze()).item()
        if id == eos_id:
            break
        word = data_layer.get_word_by_id(id)
        answer += word
        decoder_input = torch.LongTensor(np.array([id], dtype=np.int64)).unsqueeze(0)
        if torch.cuda.is_available():
            decoder_input = decoder_input.cuda()
    print(answer)
    return answer

question = "什么是ai"
model = Seq2seq(batch_size, hidden_size, vocab_size, embedding_length, False)
model.load_state_dict(torch.load('save_model/21_params.pkl'))
prediction(model, question, data_layer)



