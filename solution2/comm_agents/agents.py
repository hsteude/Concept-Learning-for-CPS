from torch import nn
import torch


class RNNEncoder(nn.Module):
    """Please implement me"""

    def __init__(self, input_size: int = 1,
                 hidden_size: int = 4,
                 rnn_layers=2,
                 latent_dim=5,
                 *args, **kwargs):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(input_size=input_size, batch_first=True,
                          hidden_size=hidden_size,
                          num_layers=rnn_layers).float()
        self.fc = nn.Linear(hidden_size, latent_dim)
        self.hidden_size = hidden_size

    def forward(self, x):
        # x.shape is (batch, seq_len, input_size)
        x, rnn_hidden_states = self.rnn(x)
        # hidden state of last leyer only
        lat_space = self.fc(rnn_hidden_states[-1, :, :])
        return lat_space


class Filter(nn.Module):
    def __init__(self, filt_initial_log_var: float = -10,
                 dim_lat_space: int = 5,
                 filt_num_decoders: int = 3,
                 **kwargs):
        super(Filter, self).__init__()
        self.selection_bias = nn.Parameter(torch.full(
            (filt_num_decoders, dim_lat_space), float(filt_initial_log_var)))

    def forward(self, lat_space, device):
        std = torch.exp(0.5 * self.selection_bias)
        eps = torch.randn(lat_space.shape[0], *std.shape, device=device)
        return [lat_space + std[i, :] * eps[:, i, :]
                for i in range(std.shape[0])]


class Decoder(nn.Module):
    def __init__(self, dec_num_question_inputs: int = 0,
                 enc_dim_lat_space: int = 5,
                 dec_hidden_size: int = 10,
                 dec_num_hidden_layers: int = 2,
                 dec_out_dim: int = 6,
                 **kwargs):
        super(Decoder, self).__init__()

        self.fc_in = nn.Linear(enc_dim_lat_space + dec_num_question_inputs,
                               dec_hidden_size)
        self.fc_hidden = nn.ModuleList(
            [nn.Linear(dec_hidden_size, dec_hidden_size)
             for i in range(dec_num_hidden_layers)])
        self.fc_out = nn.Linear(dec_hidden_size, dec_out_dim)

    def forward(self, lat_space):
        # input = torch.cat((lat_space, questions.view(-1, 1)), axis=1)
        input = lat_space
        output = torch.tanh(self.fc_in(input))
        for h in self.fc_hidden:
            output = torch.tanh(h(output))
        output = self.fc_out(output)
        return output
