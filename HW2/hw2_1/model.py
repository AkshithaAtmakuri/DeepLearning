import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import time
import torch.nn.functional as F
import numpy as np
import random
import os
from scipy.special import expit
import pickle
pickel_file='Model/picket_data.pickle'
with open(pickel_file, 'rb') as f:
        picketdata = pickle.load(f)
class Attention(nn.Module):
    def __init__(self, hidden_size):
        super(Attention, self).__init__()

        self.hidden_size = hidden_size
        self.linear1 = nn.Linear(2 * hidden_size, hidden_size)
        self.linear2 = nn.Linear(hidden_size, hidden_size)
        self.linear3 = nn.Linear(hidden_size, hidden_size)
        self.to_weight = nn.Linear(hidden_size, 1, bias=False)

    def forward(self, hidden_state, encoder_outputs):
        b_size, sequence_len, feature_n = encoder_outputs.size()
        hidden_state = hidden_state.view(b_size, 1, feature_n).repeat(1, sequence_len, 1)
        matching_inputs = torch.cat((encoder_outputs, hidden_state), 2).view(-1, 2 * self.hidden_size)
        x = self.linear1(matching_inputs)
        x = self.linear2(x)
        x = self.linear3(x)
        attention_weights = self.to_weight(x)
        attention_weights = attention_weights.view(b_size, sequence_len)
        attention_weights = F.softmax(attention_weights, dim=1)
        context = torch.bmm(attention_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context
class EncoderNet(nn.Module):
    def __init__(self):
        super(EncoderNet, self).__init__()
        self.compress = nn.Linear(4096, 512)
        self.dropout = nn.Dropout(0.2)
        self.lstm = nn.LSTM(512, 512, batch_first=True)
    def forward(self, input):
        batch_size, seq_len, feat_n = input.size()
        data_in = input.view(-1, feat_n)
        data_in = self.compress(data_in)
        data_in = self.dropout(data_in)
        data_in = data_in.view(batch_size, seq_len, 512)
        output, t = self.lstm(data_in)
        hidden_state, context = t[0], t[1]
        return output, hidden_state
class DecoderNet(nn.Module):
    def __init__(self, hidden_size, output_size, vocab_size, word_dim, dropout_percentage=0.3):
        super(DecoderNet, self).__init__()
        self.hidden_size = 512
        self.output_size = len(picketdata) + 4
        self.vocab_size = len(picketdata) + 4
        self.word_dim = 1024
        self.embedding = nn.Embedding(len(picketdata) + 4, 1024)
        self.dropout = nn.Dropout(0.3)
        self.lstm = nn.LSTM(hidden_size + word_dim, hidden_size, batch_first=True)
        self.attention = Attention(hidden_size)
        self.to_final_output = nn.Linear(hidden_size, output_size)
    def forward(self, encoder_last_hidden_state, encoder_output, targets=None, mode='train', tr_steps=None):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_chs = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_cxt = torch.zeros(decoder_chs.size())
        decoder_cxt = decoder_cxt
        decoder_ciw = Variable(torch.ones(batch_size, 1)).long()
        decoder_ciw = decoder_ciw
        seq = []
        seq_predictions = []
        targets = self.embedding(targets)
        _, seq_len, _ = targets.size()
        for i in range(seq_len - 1):
            threshold = self.helper(training_steps=tr_steps)
            if random.uniform(0.05, 0.995) > threshold:  # returns a random float value between 0.05 and 0.995
                current_input_word = targets[:, i]
            else:
                current_input_word = self.embedding(decoder_ciw).squeeze(1)
            context = self.attention(decoder_chs, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_chs, decoder_cxt))
            decoder_chs, decoder_cxt = t[0], t[1]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq.append(logprob.unsqueeze(1))
            decoder_ciw = logprob.unsqueeze(1).max(2)[1]
        seq = torch.cat(seq, dim=1)
        seq_predictions = seq.max(2)[1]
        return seq, seq_predictions
    def infer(self, encoder_last_hidden_state, encoder_output):
        _, batch_size, _ = encoder_last_hidden_state.size()
        decoder_chs = None if encoder_last_hidden_state is None else encoder_last_hidden_state
        decoder_ciw = Variable(torch.ones(batch_size, 1)).long()
        decoder_ciw = decoder_ciw
        decoder_c= torch.zeros(decoder_chs.size())
        decoder_c = decoder_c
        seq = []
        seq_predictions = []
        assumption_seq_len = 28
        for i in range(assumption_seq_len - 1):
            current_input_word = self.embedding(decoder_ciw).squeeze(1)
            context = self.attention(decoder_chs, encoder_output)
            lstm_input = torch.cat([current_input_word, context], dim=1).unsqueeze(1)
            lstm_output, t = self.lstm(lstm_input, (decoder_chs, decoder_c))
            decoder_chs, decoder_c = t[0], t[1]
            logprob = self.to_final_output(lstm_output.squeeze(1))
            seq.append(logprob.unsqueeze(1))
            decoder_ciw = logprob.unsqueeze(1).max(2)[1]
        seq = torch.cat(seq, dim=1)
        seq_predictions = seq.max(2)[1]
        return seq, seq_predictions
    def helper(self, training_steps):
        return (expit(training_steps / 20 + 0.85))
class ModelMain(nn.Module):
    def __init__(self, encoder, decoder):
        super(ModelMain, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
    def forward(self, avi_feat, mode, target_sentences=None, tr_steps=None):
        en_out, en_lhs = self.encoder(avi_feat)
        if mode == 'train':
            seq_1, seq_2= self.decoder(encoder_last_hidden_state=en_lhs,
                                       encoder_output=en_out,
                                       targets=target_sentences, mode=mode, tr_steps=tr_steps)
        elif mode == 'inference':
            seq_1, seq_2 = self.decoder.infer(encoder_last_hidden_state=en_lhs,
                                              encoder_output=en_out)
        return seq_1, seq_2

