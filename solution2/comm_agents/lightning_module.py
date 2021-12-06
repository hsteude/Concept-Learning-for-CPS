import torch
import pytorch_lightning as pl
from comm_agents.agents import RNNEncoder, Decoder, Filter
from loguru import logger


class LitModule(pl.LightningModule):
    def __init__(self,
                 enc_input_size=3,
                 enc_hidden_size=20,
                 enc_rnn_layers=4,
                 latent_dim=5,
                 filt_initial_log_var=-10,
                 filt_num_decoders=4,
                 dec_num_question_inputs=0,
                 dec_hidden_size=10,
                 dec_num_hidden_layers=6,
                 dec_out_dim=1,
                 beta: float = 0.001,
                 pretrain_thres: float = 0.001,
                 learning_rate=1e-4,
                 *args, ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.filt_num_decoders = filt_num_decoders
        self.pretrain = True
        self.beta = beta
        self.pretrain_thres = pretrain_thres
        self.latent_dim = latent_dim
        self.filt_initial_log_var = filt_initial_log_var

        # encoder init
        self.encoder_agent = RNNEncoder(input_size=enc_input_size,
                                        hidden_size=enc_hidden_size,
                                        rnn_layers=enc_rnn_layers,
                                        latent_dim=latent_dim)

        # decoder init
        # the following is ugly. i did it this way, bcause only attributes of
        # type nn.Module will get send to GPUs (e.g. lists of nn.Modules won't)
        dec_names = [f'dec_{d}' for d in range(self.filt_num_decoders)]
        for dn in dec_names:
            setattr(self, dn, Decoder(
                dec_num_question_inputs=dec_num_question_inputs,
                enc_dim_lat_space=latent_dim,
                dec_hidden_size=dec_hidden_size,
                dec_num_hidden_layers=dec_num_hidden_layers,
                dec_out_dim=dec_out_dim))
        self.decoding_agents = [getattr(self, dn) for dn in dec_names]

        # filter init
        self.filter = Filter(filt_initial_log_var=filt_initial_log_var,
                             filt_num_decoders=filt_num_decoders)

    def forward(self, x):
        lat_space = self.encoder_agent(x)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = torch.cat(dec_outs, axis=1)
        return lat_space, dec_outs

    def loss_function(self, dec_outs, answers, selection_bias, beta):
        mse_loss = torch.nn.MSELoss()
        answer_loss = mse_loss(dec_outs, answers)
        filter_loss = torch.mean(-torch.sum(selection_bias, axis=1))
        return answer_loss + beta * filter_loss

    def training_step(self, batch, batch_idx):
        x, answers, _, _ = batch
        lat_space = self.encoder_agent(x)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = torch.cat(dec_outs, axis=1)

        # set beta to 0 and force selection bias to initial value
        # if within pre-training phase (see validation step for phase switch)
        if self.pretrain:
            with torch.no_grad():
                self.filter.selection_bias[:, :] = \
                    torch.empty(*self.filter.selection_bias.shape)\
                    .fill_(self.filt_initial_log_var)
        beta = 0 if self.pretrain else self.beta

        loss = self.loss_function(dec_outs, answers,
                                  self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        self.log_selection_biases()
        return loss

    def validation_step(self, batch, batch_idx):
        x, answers, _, _ = batch
        lat_space = self.encoder_agent(x)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = [dec(lat_space) for dec in self.decoding_agents]
        dec_outs = torch.cat(dec_outs, axis=1)
        beta = 0 if self.pretrain else self.beta
        val_loss = self.loss_function(dec_outs, answers,
                                      self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

        # phase switch
        if val_loss < self.pretrain_thres:
            logger.info('setting beta to False')
            self.pretrain = False
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def log_selection_biases(self):
        """Logs the selection bias for each agent to tensorboard"""
        for i in range(self.filt_num_decoders):
            self.logger.experiment.add_scalars(
                f'selection_bias_dec{i}',
                {f'lat_neu{j}': self.filter.selection_bias[i, j]
                    for j in range(self.latent_dim)},
                global_step=self.global_step)




## old
import torch
import pytorch_lightning as pl
from comm_agents.agents import RNNEncoder, Decoder, Filter


class LitModule(pl.LightningModule):
    def __init__(self,
                 enc_input_size=3,
                 enc_hidden_size=20,
                 enc_rnn_layers=4,
                 latent_dim=5,
                 filt_initial_log_var=-10,
                 filt_num_decoders=4,
                 dec_num_question_inputs=0,
                 dec_hidden_size=10,
                 dec_num_hidden_layers=6,
                 dec_out_dim=1,
                 beta: float = 0.001,
                 pretrain_thres: float = 0.001,
                 learning_rate=1e-3,
                 *args, ** kwargs):
        super(LitModule, self).__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.filt_num_decoders = filt_num_decoders
        self.pretrain = True
        self.beta = beta
        self.pretrain_thres = pretrain_thres
        self.latent_dim = latent_dim
        self.filt_initial_log_var = filt_initial_log_var

        # encoder init
        self.encoder_agent = RNNEncoder(input_size=enc_input_size,
                                        hidden_size=enc_hidden_size,
                                        rnn_layers=enc_rnn_layers,
                                        latent_dim=latent_dim)

        # decoder init
        # the following is ugly. i did it this way, bcause only attributes of
        # type nn.Module will get send to GPUs (e.g. lists of nn.Modules won't)
        dec_names = [f'dec_{d}' for d in range(self.filt_num_decoders)]
        for dn in dec_names:
            setattr(self, dn, Decoder(
                dec_num_question_inputs=dec_num_question_inputs,
                enc_dim_lat_space=latent_dim,
                dec_hidden_size=dec_hidden_size,
                dec_num_hidden_layers=dec_num_hidden_layers,
                dec_out_dim=dec_out_dim))
        self.decoding_agents = [getattr(self, dn) for dn in dec_names]

        # filter init
        self.filter = Filter(filt_initial_log_var=filt_initial_log_var,
                             filt_num_decoders=filt_num_decoders)

    def forward(self, x):
        out = self.encoder_agent(x)
        return out

    def loss_function(self, dec_outs, answers, selection_bias, beta):
        mse_loss = torch.nn.MSELoss()
        answer_loss = mse_loss(dec_outs, answers)
        filter_loss = torch.mean(-torch.sum(selection_bias, axis=1))
        return answer_loss + beta * filter_loss

    def training_step(self, batch, batch_idx):
        x, answers, _, _ = batch
        lat_space = self.encoder_agent(x)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = torch.cat(dec_outs, axis=1)

        # set beta to 0 and force selection bias to initial value
        # if within pre-training phase (see validation step for phase switch)
        if self.pretrain:
            with torch.no_grad():
                self.filter.selection_bias[:, :] = \
                    torch.empty(*self.filter.selection_bias.shape)\
                    .fill_(self.filt_initial_log_var)
        beta = 0 if self.pretrain else self.beta

        loss = self.loss_function(dec_outs, answers,
                                  self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"train_loss": loss})
        self.log_selection_biases()
        return loss

    def validation_step(self, batch, batch_idx):
        x, answers, _, _ = batch
        lat_space = self.encoder_agent(x)
        lat_space_filt_ls = self.filter(lat_space, device=self.device)
        dec_outs = [dec(ls) for dec, ls in zip(
            self.decoding_agents, lat_space_filt_ls)]
        dec_outs = [dec(lat_space) for dec in self.decoding_agents]
        dec_outs = torch.cat(dec_outs, axis=1)
        beta = 0 if self.pretrain else self.beta
        val_loss = self.loss_function(dec_outs, answers,
                                      self.filter.selection_bias, beta)
        self.logger.experiment.add_scalars("losses", {"val_loss": val_loss})

        # phase switch
        if val_loss < self.pretrain_thres:
            self.pretrain = False
        return val_loss

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(),
                                lr=self.learning_rate)

    def log_selection_biases(self):
        """Logs the selection bias for each agent to tensorboard"""
        for i in range(self.filt_num_decoders):
            self.logger.experiment.add_scalars(
                f'selection_bias_dec{i}',
                {f'lat_neu{j}': self.filter.selection_bias[i, j]
                    for j in range(self.latent_dim)},
                global_step=self.global_step)
