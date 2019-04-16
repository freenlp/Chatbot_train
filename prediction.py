import util.data_load as dl
import torch
import numpy as np
from model.lstm import Seq2seq
import torch.nn.functional as F


class QaEngine:
    def __init__(self):
        # 词汇表
        vo_file = 'data/vocab.txt'

        self.use_class = True

        self.sentence_len = 80
        self.vo_size = 500
        self.batch_size = 1

        self.hidden_size = 256
        self.embedding_length = 100
        self.data_layer = dl.PredictionData(vo_file, self.sentence_len)
        self.model = Seq2seq(self.batch_size, self.hidden_size, self.vo_size, self.embedding_length, self.use_class)
        self.model.load_state_dict(torch.load('save_model/61_params.pkl'))

        if torch.cuda.is_available():
            self.model.cuda()
        self.model.eval()

    def prediction(self, question):
        result = dict()
        eos_id = self.data_layer.get_eos_id()
        encoder_input, decoder_input = self.data_layer.get_ids_by_words(question)
        encoder_input = encoder_input.unsqueeze(0)
        if torch.cuda.is_available():
            encoder_input = encoder_input.cuda()
            decoder_input = decoder_input.cuda()
        if self.use_class:
            output, hidden, pre_class = self.model.encoder(encoder_input, 1, None)
            pre_class = pre_class.squeeze()
            pre_class = F.softmax(pre_class, dim=0)
            class_id = torch.argmax(pre_class).item()
            result['class_id'] = class_id
        else:
            output, hidden = self.model.encoder(encoder_input, 1, None)
        decoder_hidden = hidden
        decoder_input = decoder_input.unsqueeze(0)
        answer = ""
        for di in range(self.sentence_len):
            decoder_output, decoder_hidden = self.model.decoder(
                decoder_input, decoder_hidden)
            id = torch.argmax(decoder_output.squeeze()).item()
            if id == eos_id:
                break
            word = self.data_layer.get_word_by_id(id)
            answer += word
            decoder_input = torch.LongTensor(np.array([id], dtype=np.int64)).unsqueeze(0)
            if torch.cuda.is_available():
                decoder_input = decoder_input.cuda()
        result['answer'] = answer
        return result

if __name__ == "__main__":

    question = "什么是ai"
    qa = QaEngine()
    answer = qa.prediction(question)
    print(answer)


