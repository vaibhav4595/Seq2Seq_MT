import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = vocab_size
        self.embed_size = embed_size
        self.dropout_rate = 0.2
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, num_layers=2)

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.LSTM(packed, None)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        # Changes for Bidi to work
        #hidden = (torch.cat((hidden[0][0], hidden[1][0]), dim=1).unsqueeze(0), torch.cat((hidden[0][1], hidden[1][1]), dim=1).unsqueeze(0))
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout_rate = 0.2
        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, num_layers=2) #TODO : for attention add hidden size to input
        self.dropout = nn.Dropout(self.dropout_rate)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, output, flag=0, output_lengths=None):
        embedded = self.embedding(output)
        #embedded = F.relu(embedded)
        embedded = self.dropout(embedded)
        output, hidden = self.LSTM(embedded, hidden)
        output = self.out(output)
        #output = F.log_softmax(self.out(hidden[0]), dim=2)
        return output, hidden
