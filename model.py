import torch
import torch.nn as nn
import torch.nn.functional as F

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.input_size = vocab_size
        self.embed_size = embed_size

        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size)

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        import pdb; pdb.set_trace()
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.LSTM(packed, None)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        return output, hidden

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size

        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size) #TODO : for attention add hidden size to input
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, hidden, output):
        embedded = self.embedding(output)
        output, hidden = self.LSTM(output, hidden)
        output = F.softmax(self.out(hidden))
        return output, hidden
