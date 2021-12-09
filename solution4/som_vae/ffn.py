from torch import nn


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
        # we want the shape (batch_size, seq_len, num_sensors)
        return out.reshape(-1, self.seq_len, self.input_size)