import torch
from torch import nn
import pytorch_lightning as pl
from solution3.ae_sindy.sindy_library import SINDyLibrary

pl.seed_everything(6542198)


class Encoder(nn.Module):
    def __init__(
        self,
        input_dim=10000,
        hidden_sizes=[2048, 512, 128, 64],
        enc_out_dim=3,
        activation=nn.ReLU(),
        *args,
        **kwargs,
    ):
        super(Encoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], enc_out_dim)
        self.activation = activation
        self.initialize_weights()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.fc5(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class Decoder(nn.Module):
    def __init__(
        self,
        input_dim=3,
        hidden_sizes=[64, 128, 512, 2048],
        dec_output_dim=10000,
        activation=nn.ReLU(),
        *args,
        **kwargs,
    ):
        super(Decoder, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_sizes[0])
        self.fc2 = nn.Linear(hidden_sizes[0], hidden_sizes[1])
        self.fc3 = nn.Linear(hidden_sizes[1], hidden_sizes[2])
        self.fc4 = nn.Linear(hidden_sizes[2], hidden_sizes[3])
        self.fc5 = nn.Linear(hidden_sizes[3], dec_output_dim)
        self.activation = activation
        self.initialize_weights()

    def forward(self, x):
        out = self.activation(self.fc1(x))
        out = self.activation(self.fc2(out))
        out = self.activation(self.fc3(out))
        out = self.activation(self.fc4(out))
        out = self.fc5(out)
        return out

    def initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.xavier_normal_(m.weight)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)


class SINDyAutoencoder(pl.LightningModule):
    def __init__(
        self,
        learning_rate=0.001,
        input_dim=10000,
        latent_dim=3,
        activation="relu",
        enc_hidden_sizes=[2048, 512, 128, 64],
        dec_hidden_sizes=[64, 128, 512, 2048],
        sindy_biases=True,
        sindy_states=False,
        sindy_sin=False,
        sindy_cos=False,
        sindy_multiply_pairs=True,
        sindy_poly_order=3,
        sindy_sqrt=False,
        sindy_inverse=False,
        sindy_sign_sqrt_of_diff=True,
        XI_fill_val=0.0,
        sequential_thresholding=True,
        sequential_thresholding_freq=10,
        sequential_thresholding_thres=1e-4,
        loss_weight_sindy_x=1,
        loss_weight_sindy_z=1,
        loss_weight_sindy_regularization=1,
        adam_eps=1e-4,
        *args,
        **kwargs,
    ):
        super().__init__()
        self.learning_rate = learning_rate
        self.adam_eps = adam_eps
        self.save_hyperparameters()
        self.num_layers = 3
        self.activation_function_str = activation
        if activation == "relu":
            self.activation_function = nn.ReLU()
        elif activation == "sigmoid":
            self.activation_function = nn.Sigmoid()
        elif activation == "tanh":
            self.activation_function = nn.Tanh()
        else:
            print("nooooo!")

        self.phi_x = Encoder(
            input_dim=input_dim,
            hidden_sizes=enc_hidden_sizes,
            enc_out_dim=latent_dim,
            activation=self.activation_function,
        )
        self.psi_z = Decoder(
            input_dim=latent_dim,
            hidden_sizes=dec_hidden_sizes,
            dec_output_dim=input_dim,
            activation=self.activation_function,
        )
        self.SINDyLibrary = SINDyLibrary(
            device="cuda:0",
            latent_dim=latent_dim,
            include_biases=sindy_biases,
            include_states=sindy_states,
            include_sin=sindy_sin,
            include_cos=sindy_cos,
            include_multiply_pairs=sindy_multiply_pairs,
            poly_order=sindy_poly_order,
            include_sqrt=sindy_sqrt,
            include_inverse=sindy_inverse,
            include_sign_sqrt_of_diff=sindy_sign_sqrt_of_diff,
        )

        self.XI = nn.Parameter(
            torch.full(
                (self.SINDyLibrary.number_candidate_functions, latent_dim), XI_fill_val
            )
        )

        self.sequential_thresholding = sequential_thresholding
        self.sequential_thresholding_freq = sequential_thresholding_freq
        self.sequential_thresholding_thres = sequential_thresholding_thres
        self.XI_coefficient_mask = torch.ones(
            (self.SINDyLibrary.number_candidate_functions, latent_dim), device="cuda:0"
        )

        self.loss_weight_sindy_x = loss_weight_sindy_x
        self.loss_weight_sindy_z = loss_weight_sindy_z
        self.loss_weight_sindy_regularization = loss_weight_sindy_regularization
        self.loss_names = [
            "total_loss",
            "recon_loss",
            "sindy_loss_x",
            "sindy_loss_z",
            "sindy_regular_loss",
            "z_real_loss",
        ]

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=self.learning_rate)

    def loss_function(self, x, xdot, x_hat, xdot_hat, zdot, zdot_hat, XI):
        mse = nn.MSELoss()
        recon_loss = mse(x, x_hat)
        sindy_loss_x = mse(xdot, xdot_hat)
        sindy_loss_z = mse(zdot, zdot_hat)
        sindy_regular_loss = torch.sum(torch.abs(XI))
        loss = (
            recon_loss
            + self.loss_weight_sindy_x * sindy_loss_x
            + self.loss_weight_sindy_z * sindy_loss_z
            + self.loss_weight_sindy_regularization * sindy_regular_loss
        )
        return (
            loss,
            recon_loss,
            sindy_loss_x,
            sindy_loss_z,
            sindy_regular_loss,
        )

    def compute_nn_derivates_wrt_time(
        self, y, ydot, weights_list, biases_list, activation="sigmoid"
    ):
        input_l = y
        d_input_l = ydot
        if activation == "sigmoid":
            for l in range(len(weights_list) - 1):
                input_l = torch.matmul(input_l, weights_list[l].T) + biases_list[l]
                input_l = torch.sigmoid(input_l)
                d_input_l = torch.mul(
                    torch.mul(input_l, 1 - input_l),  # outer
                    torch.matmul(d_input_l, weights_list[l].T),  # outer  # inner
                )
            d_input_l = torch.matmul(d_input_l, weights_list[-1].T)

        if activation == "tanh":
            for l in range(len(weights_list) - 1):
                input_l = torch.matmul(input_l, weights_list[l].T) + biases_list[l]
                input_l = torch.tanh(input_l)
                d_input_l = torch.mul(
                    1 - torch.square(torch.tanh(input_l)),  # outer
                    torch.matmul(d_input_l, weights_list[l].T),  # outer  # inner
                )
            d_input_l = torch.matmul(d_input_l, weights_list[-1].T)
        return d_input_l

    def forward(self, x, xdot):
        return self._shared_eval(x, xdot)

    def _shared_eval(self, x, xdot):
        z = self.phi_x(x)
        x_hat = self.psi_z(z)
        theta = self.SINDyLibrary.transform(z)
        if self.sequential_thresholding:
            zdot_hat = torch.matmul(theta, self.XI_coefficient_mask * self.XI)
        else:
            zdot_hat = torch.matmul(theta, self.XI)
        phi_x_parameters = list(self.phi_x.parameters())
        phi_x_weight_list = [w for w in phi_x_parameters if len(w.shape) == 2]
        phi_x_biases_list = [b for b in phi_x_parameters if len(b.shape) == 1]
        zdot = self.compute_nn_derivates_wrt_time(
            x,
            xdot,
            phi_x_weight_list,
            phi_x_biases_list,
            activation=self.activation_function_str,
        )
        psi_z_parameters = list(self.psi_z.parameters())
        psi_z_weight_list = [w for w in psi_z_parameters if len(w.shape) == 2]
        psi_z_biases_list = [b for b in psi_z_parameters if len(b.shape) == 1]
        xdot_hat = self.compute_nn_derivates_wrt_time(
            z,
            zdot_hat,
            psi_z_weight_list,
            psi_z_biases_list,
            activation=self.activation_function_str,
        )
        return x_hat, xdot_hat, z, zdot, zdot_hat

    def training_step(self, batch, batch_idx):
        x, xdot, _, _, _ = batch
        x_hat, xdot_hat, _, zdot, zdot_hat = self._shared_eval(x, xdot)
        losses = self.loss_function(
            x=x,
            xdot=xdot,
            x_hat=x_hat,
            xdot_hat=xdot_hat,
            zdot=zdot,
            zdot_hat=zdot_hat,
            XI=self.XI,
        )
        loss_dict = dict(zip([f"train_{n}" for n in self.loss_names], losses))
        self.logger.experiment.add_scalars("loss", loss_dict)
        return loss_dict["train_total_loss"]

    def validation_step(self, batch, batch_idx):
        x, xdot, _, _, _ = batch
        x_hat, xdot_hat, _, zdot, zdot_hat = self._shared_eval(x, xdot)
        losses = self.loss_function(
            x=x,
            xdot=xdot,
            x_hat=x_hat,
            xdot_hat=xdot_hat,
            zdot=zdot,
            zdot_hat=zdot_hat,
            XI=self.XI,
        )
        loss_dict = dict(zip([f"val_{n}" for n in self.loss_names], losses))
        self.logger.experiment.add_scalars("loss", loss_dict)
        return loss_dict["val_total_loss"]
