import torch
from torch import nn
import pytorch_lightning as pl
from loguru import logger
import numpy as np
from pytorch_lightning.callbacks import Callback

pl.seed_everything(16548)


class RNNEncoder(nn.Module):
    """Please implement me"""

    def __init__(
        self, input_size: int = 1, hidden_size: int = 4, rnn_layers=5, *args, **kwargs
    ):
        super(RNNEncoder, self).__init__()
        self.rnn = nn.GRU(
            input_size=input_size,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        ).float()
        self.hidden_size = hidden_size

    def forward(self, x):
        # x.shape is (batch, seq_len, input_size)
        x, rnn_hidden_states = self.rnn(x)
        final_hidden_state = rnn_hidden_states[-1, :, :]
        # to be consistent with the linear encoder we want the output shape to be
        # (batch_size, seq_len*input_size)
        return final_hidden_state


class RNNDecoder(nn.Module):
    """Please write me"""

    def __init__(
        self,
        latent_dim=4,
        hidden_size=10,
        rnn_layers=5,
        input_size=2,
        seq_len=100,
        *args,
        **kwargs,
    ):
        super(RNNDecoder, self).__init__()
        self.rnn1 = nn.GRU(
            input_size=latent_dim,
            batch_first=True,
            hidden_size=hidden_size,
            num_layers=rnn_layers,
        ).float()
        # self.rnn2 = nn.GRU(input_size=hidden_size, batch_first=True,
        # hidden_size=input_size, num_layers=rnn_layers).float()
        self.fc = nn.Linear(in_features=hidden_size, out_features=input_size)
        self.seq_len = seq_len

    def forward(self, x):
        # so x has the shape (batch, seq_lwn, input_size)
        x = x.reshape(-1, 1, x.shape[-1]).repeat(1, self.seq_len, 1)
        x, _ = self.rnn1(x)
        # another fc layer to alower for values out of range [-1,1]
        out = self.fc(x)
        # output should have shape (batch_size, seq_len, num_sensors)
        return out


class MLPEncoder(nn.Module):
    def __init__(self, input_size=2, seq_len=30, hidden_size=100, *args, **kwargs):
        super(MLPEncoder, self).__init__()
        self.fc1 = nn.Linear(seq_len * input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, hidden_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        x_in = x.reshape(x.shape[0], -1)
        out = self.relu(self.fc1(x_in))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.tanh(self.fc4(out))
        # output shape should be: (batch_size, seq_len * num_sensors)
        return out


class MLPDecoder(nn.Module):
    def __init__(
        self, latent_dim=3, hidden_size=100, seq_len=30, input_size=2, *args, **kwargs
    ):
        super(MLPDecoder, self).__init__()
        self.input_size = input_size
        self.seq_len = seq_len
        self.fc1 = nn.Linear(latent_dim, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, hidden_size)
        self.fc4 = nn.Linear(hidden_size, seq_len * input_size)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

    def forward(self, x):
        out = self.relu(self.fc1(x))
        out = self.relu(self.fc2(out))
        out = self.relu(self.fc3(out))
        out = self.fc4(out)
        # as in the rnn decoder / encoder we want the shape
        # (batch_size, seq_len, num_sensors)
        return out.reshape(-1, self.seq_len, self.input_size)


class VAE(pl.LightningModule):
    def __init__(
        self,
        enc_out_dim=4,
        learning_rate=1e-3,
        latent_dim=4,
        input_size=2,
        seq_len=100,
        beta=10,
        hidden_size=10,
        rnn_layers=1,
        rnn_models=True,
        *args,
        **kwargs,
    ):
        super().__init__()

        self.save_hyperparameters()

        # encoder, decoder
        if rnn_models:
            self.encoder = RNNEncoder(
                input_size=input_size, hidden_size=hidden_size, rnn_layers=rnn_layers
            )
            self.decoder = RNNDecoder(
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                rnn_layers=rnn_layers,
                input_size=input_size,
                seq_len=seq_len,
            )
        else:
            self.encoder = MLPEncoder(
                input_size=input_size, hidden_size=hidden_size, seq_len=seq_len
            )
            self.decoder = MLPDecoder(
                latent_dim=latent_dim,
                hidden_size=hidden_size,
                input_size=input_size,
                seq_len=seq_len,
            )

        # distribution parameters
        self.fc_mu = nn.Linear(enc_out_dim, latent_dim)
        self.fc_var = nn.Linear(enc_out_dim, latent_dim)

        # for the gaussian likelihood
        self.log_scale_diag = nn.Parameter(torch.zeros(seq_len * input_size))

        # for beta term of beta-variational autoencoder
        self.beta = beta
        self.learning_rate = learning_rate

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    @staticmethod
    def gaussian_likelihood(x_hat, logscale_diag, x):
        scale_diag = torch.exp(logscale_diag)
        scale = torch.diag(scale_diag)
        mu_x = x_hat.reshape(x_hat.shape[0], -1)
        dist = torch.distributions.MultivariateNormal(mu_x, scale_tril=scale)

        # measure prob of seeing x under p(x|z)
        log_pxz = dist.log_prob(x.reshape(mu_x.shape[0], -1))
        return log_pxz

    @staticmethod
    def kl_divergence(mu, std):

        # 1. define the first two probabilities (in this case Normal for both)
        p = torch.distributions.Normal(torch.zeros_like(mu), torch.ones_like(std))
        q = torch.distributions.Normal(mu, std)

        kl = torch.distributions.kl.kl_divergence(q, p)
        kl = kl.sum(-1)

        return kl

    def forward(self, x):
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        return mu, std, z, x_hat, self.log_scale_diag

    def _shared_eval(self, x):

        # encode x to get the mu and variance parameters
        x_encoded = self.encoder(x)
        mu, log_var = self.fc_mu(x_encoded), self.fc_var(x_encoded)

        # sample z from q
        std = torch.exp(log_var / 2)
        q = torch.distributions.Normal(mu, std)
        z = q.rsample()

        # decoded
        x_hat = self.decoder(z)

        # reconstruction loss
        recon_loss = self.gaussian_likelihood(x_hat, self.log_scale_diag, x)

        # kl
        kl = self.kl_divergence(mu, std)

        # elbo
        elbo = self.beta * kl - recon_loss
        elbo = elbo.mean()

        log_dict = dict(
            {
                "elbo": elbo,
                "kl": kl.mean(),
                "recon_loss": recon_loss.mean(),
            }
        )

        return elbo, log_dict

    def training_step(self, batch, batch_idx):
        x, _, _ = batch
        train_elbo, train_log_dict = self._shared_eval(x)
        self.logger.experiment.add_scalars(
            "loss",
            dict(
                elbo_train=train_log_dict["elbo"],
                kl_train=train_log_dict["kl"],
                recon_loss_train=train_log_dict["recon_loss"],
                beta=self.beta,
            ),
        )
        return train_elbo

    def validation_step(self, batch, batch_idx):
        x, _, _ = batch
        self._shared_eval(x)
        val_elbo, val_log_dict = self._shared_eval(x)
        self.logger.experiment.add_scalars(
            "loss",
            dict(
                elbo_val=val_log_dict["elbo"],
                kl_val=val_log_dict["kl"],
                recon_loss_val=val_log_dict["recon_loss"],
            ),
        )

        return val_elbo


class BetaIncreaseCallBack(Callback):
    def __init__(self, initial_beta, beta_max, increase_after_n_epochs, number_steps):
        super().__init__()
        self.idx = 0
        self.num_epochs = 100
        self.betas = [initial_beta] * 4 + list(
            np.linspace(initial_beta, beta_max, number_steps)
        )

    def on_train_epoch_end(self, trainer, pl_modelu):
        if self.idx < len(self.betas):
            pl_modelu.beta = self.betas[self.idx]
            if trainer.current_epoch % self.num_epochs == 0:
                if self.idx <= len(self.betas):
                    logger.info(f"Set Beta to {self.betas[self.idx]}")
                    self.idx += 1
