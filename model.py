import torch
import torch.nn as nn
import torch.nn.functional as F
from pdb import set_trace as bp

class EncoderRNN(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size):
        super(EncoderRNN, self).__init__()

        self.hidden_size = hidden_size // 2
        self.input_size = vocab_size
        self.embed_size = embed_size
        self.dropout_rate = 0.2
        self.num_layers = 2
        self.embedding = nn.Embedding(self.input_size, self.embed_size)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.LSTM = nn.LSTM(self.embed_size, self.hidden_size, num_layers=self.num_layers, dropout=self.dropout_rate, bidirectional=True)
        self.bidi = True

    def forward(self, input, input_lengths):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)
        packed = torch.nn.utils.rnn.pack_padded_sequence(embedded, input_lengths)
        output, hidden = self.LSTM(packed, None)
        output, _ = torch.nn.utils.rnn.pad_packed_sequence(output)
        if self.bidi != True:
            return output, hidden
        else:
            hidden_final = torch.cat((hidden[0][0], hidden[0][1]), dim=1).unsqueeze(0)
            cell_final = torch.cat((hidden[1][0], hidden[1][1]), dim=1).unsqueeze(0)
            for i in range(1, self.num_layers):
                hidden_final = torch.cat((hidden_final, torch.cat((hidden[0][2*i], hidden[0][2*i + 1]), dim=1).unsqueeze(0)), dim=0)
                cell_final = torch.cat((cell_final, torch.cat((hidden[1][2*i], hidden[1][2*i + 1]), dim=1).unsqueeze(0)), dim=0)
        # Changes for Bidi to work
            return output, (hidden_final, cell_final)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, output_size):
        super(DecoderRNN, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.output_size = output_size
        self.dropout_rate = 0.2
        self.embedding = nn.Embedding(self.output_size, self.embed_size)
        self.LSTM = nn.LSTM(self.hidden_size + self.embed_size, self.hidden_size, num_layers=1, dropout=self.dropout_rate)
        self.dropout = nn.Dropout(self.dropout_rate)
        self.out = nn.Linear(self.hidden_size, self.output_size)

    def forward(self, encoder_outputs, hidden, output, flag=0, output_lengths=None):
        embedded = self.embedding(output)
        embedded = F.relu(embedded)
        embedded = self.dropout(embedded)

        # Multiply (B x 1 x H) * (B x H x S) = (B x 1 x S)
        cur_hidden = hidden[0].transpose(0,1)
        encoder_hiddens = encoder_outputs.transpose(0,1).transpose(1,2)
        attn_weights = F.softmax(cur_hidden.bmm(encoder_hiddens), dim=2)

        # Determine encoder context, (B x 1 x S) * (B x S x H) = (B x 1 x H)
        encoder_contexts = attn_weights.bmm(encoder_outputs.transpose(0,1))

        # Concate with embedded
        rnn_input = torch.cat((embedded, encoder_contexts.transpose(0,1)), dim=2)

        output, hidden = self.LSTM(rnn_input, hidden)
        output = self.out(output)
        #output = F.log_softmax(self.out(hidden[0]), dim=2)
        return output, hidden
