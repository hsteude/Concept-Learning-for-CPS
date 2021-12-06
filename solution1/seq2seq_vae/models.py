class RNNEncoder(nn.Module):
    """Please implement me"""
    def __init__(self, input_size: int = 1,
                 hidden_size: int = 4,
                 rnn_layers=5,
                 *args, **kwargs):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.LSTM(input_size=input_size, batch_first=True,
                              hidden_size=hidden_size, num_layers=rnn_layers).float()
        self.hidden_size = hidden_size

    def forward(self, x):
        # x.shape is (batch, seq_len, input_size)
        x, (rnn_hidden_states, _) = self.rnn(x)
        final_hidden_state = rnn_hidden_states[-1, :, :]
        # to be consistent with the linear encoder we want the output shape to be
        # (batch_size, seq_len*input_size)
        return final_hidden_state


class RNNDecoder(nn.Module):
    """Please write me"""
    def __init__(self,
                 latent_dim=4,
                 hidden_size=10,
                 rnn_layers=5,
                 input_size=2,
                 seq_len=100, *args, **kwargs):
        super(RNNDecoder, self).__init__()
        self.rnn1 = nn.LSTM(input_size=latent_dim, batch_first=True,
                              hidden_size=hidden_size, num_layers=rnn_layers).float()
        self.rnn2 = nn.LSTM(input_size=hidden_size, batch_first=True,
                              hidden_size=input_size, num_layers=rnn_layers).float()
        self.fc = nn.Linear(in_features=input_size, out_features=input_size)
        self.seq_len = seq_len

    def forward(self, x):
        # so x has the shape (batch, seq_lwn, input_size)
        x = x.reshape(-1, 1, x.shape[-1]).repeat(1, self.seq_len, 1)
        x, (_, _) = self.rnn1(x)
        x, (_, _) = self.rnn2(x)
        # another fc layer to alower for values out of range [-1,1]
        out = self.fc(x)
        # output should have shape (batch_size, seq_len, num_sensors)
        return out

class LinearEncoder(nn.Module):
    def __init__(self, input_size=2, seq_len=30, hidden_size=100, enc_out_dim=30, *args, **kwargs):
        super(LinearEncoder, self).__init__()
        self.fc1 = nn.Linear(seq_len*input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, enc_out_dim)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_in = x.reshape(x.shape[0], -1)
        out = self.relu(self.fc1(x_in))
        out = self.relu(self.fc2(out))
        out = self.tanh(self.fc3(out))
        # output shape should be: (batch_size, seq_len * num_sensors)
        return out

class LinearDecoder(nn.Module):
    def __init__(self, latent_dim=3, hidden_size=100, seq_len=30,
                 input_size=2, *args, **kwargs):
        super(LinearDecoder, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, seq_len*input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.fc3(out)
        # as in the rnn decoder / encoder we want the shape (batch_size, seq_len, num_sensors)
        return out.reshape(-1, self.seq_len, self.input_size)




