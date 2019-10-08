import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class RNNModel(nn.Module):
    """Container module with an encoder, a recurrent module, and a decoder."""

    def __init__(self, rnn_type, ntoken, ninp, nhid, nlayers, \
            dropout=0.5, tie_weights=False):
        super(RNNModel, self).__init__()
        self.drop = nn.Dropout(dropout)
        self.encoder = nn.Embedding(ntoken, ninp)
        self.decoder = nn.Linear(nhid, ntoken)

        if rnn_type in ['LSTM', 'GRU']:
            self.rnn = getattr(nn, rnn_type)(ninp, nhid, nlayers, \
                    dropout=dropout, batch_first=True)
        else:
            try:
                nonlinearity = {'RNN_TANH': 'tanh', \
                        'RNN_RELU': 'relu'}[rnn_type]
            except KeyError:
                raise ValueError( """An invalid option for `--model` was 
                supplied, options are 
                ['LSTM', 'GRU', 'RNN_TANH' or 'RNN_RELU']""")
            self.rnn = nn.RNN(ninp, nhid, nlayers, nonlinearity=nonlinearity, \
                    dropout=dropout, batch_first=True)

        # Optionally tie weights as in:
        # "Using the Output Embedding to Improve Language Models" 
        # (Press & Wolf 2016)
        # https://arxiv.org/abs/1608.05859
        # and
        # "Tying Word Vectors and Word Classifiers: A Loss Framework for 
        #      Language Modeling" (Inan et al. 2016)
        # https://arxiv.org/abs/1611.01462
        if tie_weights:
            if nhid != ninp:
                raise ValueError("""When using the tied flag, 
                nhid must be equal to emsize""")
            self.decoder.weight = self.encoder.weight

        self.init_weights()
        self.rnn_type = rnn_type
        self.nhid = nhid
        self.nlayers = nlayers

    def init_weights(self):
        initrange = 0.1
        self.encoder.weight.data.uniform_(-initrange, initrange)
        self.decoder.bias.data.zero_()
        self.decoder.weight.data.uniform_(-initrange, initrange)

    def forward(self, input, hidden, lengths):
        embs = self.drop(self.encoder(input))
        packed = pack_padded_sequence(embs, lengths, batch_first=True)
        out_packed, hidden = self.rnn(packed, hidden)
        out_unpacked, len_tensor = pad_packed_sequence(out_packed, \
                batch_first=True)
        # last_outputs = last_timestep(out_unpacked, len_tensor)
        output = self.drop(out_unpacked)
        decoded = self.decoder(output)
        return decoded, hidden

    def last_timestep(self, unpacked, lengths):
        idx = (lengths - 1).view(-1, 1).expand(unpacked.size(0), \
                unpacked.size(2)).unsqueeze(1)
        return unpacked.gather(1, idx).squeeze()

    def init_hidden(self, bsz):
        weight = next(self.parameters())
        if self.rnn_type == 'LSTM':
            # first element in tuple for hidden state h, 
            # second for cell state c
            return (weight.new_zeros(self.nlayers, bsz, self.nhid),
                    weight.new_zeros(self.nlayers, bsz, self.nhid))
        else:
            return weight.new_zeros(self.nlayers, bsz, self.nhid)
