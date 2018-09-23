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

    def forward(self, hidden, output, output_lengths):
        # Embed, pack
        embedded = self.embedding(output)

	# Sort everything by output length
        sorted_indices = sorted(range(len(output_lengths)), key=lambda i: output_lengths[i], reverse=True)
        sorted_lengths = [output_lengths[i] for i in sorted_indices]
        hidden = (hidden[0][:,sorted_indices], hidden[1][:,sorted_indices])
        embedded = embedded[:, sorted_indices]

        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, sorted_lengths)

        # Pass packed data through LSTM, and unpack
        output, hidden = self.LSTM(packed, hidden)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)

        # Unsort indices
        unsort_indices = sorted(range(len(sorted_indices)), key=lambda i: sorted_indices[i]) 
        output = output[:, unsort_indices]
        hidden = (hidden[0][:,sorted_indices], hidden[1][:,sorted_indices])
	
        output = F.softmax(self.out(output), dim=2)
        return output, hidden
