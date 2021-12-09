import torch
from torch import nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import ReduceLROnPlateau

from solution4.som_vae.ffn import LinearDecoder, LinearEncoder


class SOMVAE(pl.LightningModule):
    def __init__(self, d_input=100, d_channel=3, d_enc_dec=100,
                 d_latent=64, d_som=None,
                 alpha=1, beta=1, gamma=1, tau=1,
                 lr=1e-3):
        super().__init__()
        self.save_hyperparameters()  # stores hyperparameters in self.hparams and allows logging
        self.d_som = d_som if d_som is not None else [3, 3]
        self.d_latent = d_latent

        self.encoder = LinearEncoder(input_size=d_channel, seq_len=d_input,
                                     hidden_size=d_enc_dec, enc_out_dim=d_latent)
        self.decoder_e = LinearDecoder(latent_dim=d_latent, hidden_size=d_enc_dec,
                                       seq_len=d_input, input_size=d_channel)
        self.decoder_q = LinearDecoder(latent_dim=d_latent, hidden_size=d_enc_dec,
                                       seq_len=d_input, input_size=d_channel)

        self.embeddings = nn.Parameter(nn.init.trunc_normal_(torch.empty((d_som[0], d_som[1], d_latent)),
                                                             std=0.05, a=-0.1, b=0.1))
        self.mse_loss = nn.MSELoss()
        self.probs = self._transition_probabilities()

        self.automatic_optimization = True

    def forward(self, x):
        with torch.no_grad():
            z_e = self.encoder(x)
            z_q, z_dist, k = self._find_closest_embedding(z_e, batch_size=x.shape[0])
        return k

    def _shared_step(self, x):
        # encoding
        z_e = self.encoder(x)
        # embedding
        z_q, z_dist, k = self._find_closest_embedding(z_e, batch_size=x.shape[0])
        z_q_neighbors = self._find_neighbors(z_q, k, batch_size=x.shape[0])

        x_q = self.decoder_q(z_q)
        x_e = self.decoder_e(z_e)

        loss, raw_loss = self.loss(x, x_e, x_q, z_e, z_q, z_q_neighbors)
        return loss, raw_loss

    def _transition_probabilities(self):
        probs_raw = torch.zeros(*(self.d_som + self.d_som))
        probs_pos = torch.exp(probs_raw)
        probs_sum = torch.sum(probs_pos, dim=[-1, -2], keepdim=True)
        return nn.Parameter(probs_pos / probs_sum)

    def _find_closest_embedding(self, z_e, batch_size=32):
        """Picks the closest embedding for every encoding."""
        z_dist = (z_e.unsqueeze(1).unsqueeze(2) - self.embeddings.unsqueeze(0)) ** 2
        z_dist_sum = torch.sum(z_dist, dim=-1)
        z_dist_flat = z_dist_sum.view(batch_size, -1)
        k = torch.argmin(z_dist_flat, dim=-1)
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_batch = torch.stack([k_1, k_2], dim=1)
        return self._gather_nd(self.embeddings, k_batch), z_dist_flat, k

    def _get_coordinates_from_idx(self, k):
        k_1 = torch.div(k, self.d_som[1], rounding_mode='floor')
        k_2 = k % self.d_som[1]
        return k_1, k_2

    def _find_neighbors(self, z_q, k, batch_size):
        k_1, k_2 = self._get_coordinates_from_idx(k)

        k1_not_top = k_1 < self.d_som[0] - 1
        k1_not_bottom = k_1 > 0
        k2_not_right = k_2 < self.d_som[1] - 1
        k2_not_left = k_2 > 0

        k1_up = torch.where(k1_not_top, k_1 + 1, k_1)
        k1_down = torch.where(k1_not_bottom, k_1 - 1, k_1)
        k2_right = torch.where(k2_not_right, k_2 + 1, k_2)
        k2_left = torch.where(k2_not_left, k_2 - 1, k_2)

        z_q_up = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_up_ = self._gather_nd(self.embeddings, torch.stack([k1_up, k_2], dim=1))
        z_q_up[k1_not_top == 1] = z_q_up_[k1_not_top == 1]

        z_q_down = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_down_ = self._gather_nd(self.embeddings, torch.stack([k1_down, k_2], dim=1))
        z_q_down[k1_not_bottom == 1] = z_q_down_[k1_not_bottom == 1]

        z_q_right = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_right_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_right], dim=1))
        z_q_right[k2_not_right == 1] = z_q_right_[k2_not_right == 1]

        z_q_left = torch.zeros(batch_size, self.d_latent).to(self.device)
        z_q_left_ = self._gather_nd(self.embeddings, torch.stack([k_1, k2_left], dim=1))
        z_q_left[k2_not_left == 1] = z_q_left_[k2_not_left == 1]

        return torch.stack([z_q, z_q_up, z_q_down, z_q_right, z_q_left], dim=1)

    def _loss_reconstruct(self, x, x_e, x_q):
        l_e = self.mse_loss(x, x_e)
        l_q = self.mse_loss(x, x_q)
        mse_l = l_e + l_q
        return mse_l

    def _loss_commit(self, z_e, z_q):
        commit_l = self.mse_loss(z_e, z_q)
        return commit_l

    @staticmethod
    def _loss_som(z_e, z_q_neighbors):
        z_e = z_e.detach()
        som_l = torch.mean((z_e.unsqueeze(1) - z_q_neighbors) ** 2)
        return som_l

    def loss_prob(self, k):
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old, k_1, k_2], dim=1)

        transitions_all = self._gather_nd(self.probs, k_stacked)
        prob_l = -self.hparams.gamma * torch.mean(torch.log(transitions_all))
        return prob_l

    def _loss_z_prob(self, k, z_dist_flat):
        k_1, k_2 = self._get_coordinates_from_idx(k)
        k_1_old = torch.cat([k_1[0:1], k_1[:-1]], dim=0)
        k_2_old = torch.cat([k_2[0:1], k_2[:-1]], dim=0)
        k_stacked = torch.stack([k_1_old, k_2_old], dim=1)

        out_probabilities_old = self._gather_nd(self.probs, k_stacked)
        out_probabilities_flat = out_probabilities_old.view(k.shape[0], -1)
        weighted_z_dist_prob = z_dist_flat * out_probabilities_flat
        prob_z_l = torch.mean(weighted_z_dist_prob)
        return prob_z_l

    def loss(self, x, x_e, x_q, z_e, z_q, z_q_neighbors):
        mse_l = self._loss_reconstruct(x, x_e, x_q)
        commit_l = self._loss_commit(z_e, z_q)
        som_l = self._loss_som(z_e, z_q_neighbors)
        loss = mse_l + self.hparams.alpha * commit_l + self.hparams.beta * som_l
        raw_loss = mse_l + commit_l + som_l
        return loss, raw_loss

    @staticmethod
    def _gather_nd(params, idx):
        """Similar to tf.gather_nd. Here: returns batch of params given the indices."""
        idx = idx.long()
        outputs = []
        for i in range(len(idx)):
            outputs.append(params[[idx[i][j] for j in range(idx.shape[1])]])
        outputs = torch.stack(outputs)
        return outputs

    def training_step(self, batch, batch_id):
        loss, raw_loss = self._shared_step(batch)
        self.log("train_loss", loss)
        self.log("train_loss_raw", raw_loss)
        return loss

    def validation_step(self, batch, batch_id):
        loss, raw_loss = self._shared_step(batch)
        self.log("val_loss", loss)
        self.log("val_loss_raw", raw_loss)
        return raw_loss

    def validation_epoch_end(self, outputs):
        # log hparams with val_loss as reference
        if self.logger:
            self.logger.log_hyperparams(self.hparams, {"hp/val_loss_raw": torch.min(torch.stack(outputs))})

    def test_step(self, batch, batch_id):
        loss, raw_loss = self._shared_step(batch)
        self.log("test_loss", loss)
        self.log("test_loss_raw", raw_loss)

    def configure_optimizers(self):
        optimizer_model = Adam(self.parameters(), lr=self.hparams.lr)
        scheduler = ReduceLROnPlateau(optimizer_model, factor=0.5, patience=10, min_lr=1e-5)
        return [optimizer_model], [{"scheduler": scheduler, "interval": "epoch", "monitor": "train_loss"}]
