from solution3.datagen.dataset import ThreeTankDataSet
import plotly.io as pio
import torch
from torch.utils.data import DataLoader
import solution3.constants as const
import yaml
import os
from solution3.ae_sindy.ae_sindy import SINDyAutoencoder
import pandas as pd
import numpy as np
from scipy.integrate import odeint
from plotly.subplots import make_subplots
import plotly.graph_objects as go

# initiate dataset and data loader and load batch
dataset = ThreeTankDataSet()
batch_size = 2000
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=24)
batches = iter(dataloader)

# load model
MODEL_VERSION = "version_15"
hparams_path = f"./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/hparams.yaml"
with open(hparams_path, "r") as stream:
    hparam_dct = yaml.safe_load(stream)
ckpt_file_name = os.listdir(
    f"./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/checkpoints/"
)[-1]
ckpt_file_path = (
    f"./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/checkpoints/{ckpt_file_name}"
)
model = SINDyAutoencoder.load_from_checkpoint(ckpt_file_path)


x, xdot, z_real, z_dot_real, idxs = batches.next()
x_hat, xdot_hat, z, zdot, zdot_hat = model.cuda()(x.cuda(), xdot.cuda())

XI = model.XI.cpu().detach().numpy()
XI.max()
df_XI = pd.DataFrame(XI, columns=["z0_dot_hat", "z1_dot_hat", "z2_dot_hat"])
df_XI.index = model.SINDyLibrary.feature_names


# get model and print model
thres = 0.001
from solution3.scripts.run_training import HPARAMS
thres = HPARAMS["sequential_thresholding_thres"]
round_digits = 5
XI = model.XI.detach().cpu().numpy()
indices_var = np.where(np.abs(XI) > thres)[1]
indices_expr = np.where(np.abs(XI) > thres)[0]
indices = list(zip(indices_expr, indices_var))
values = XI[np.abs(XI) > thres]
feature_names = model.SINDyLibrary.feature_names
z0_dot_expr = "+".join(
    [f"({XI[i]:.5f}*{feature_names[i[0]]})" for i in indices if i[1] == 0]
)
z1_dot_expr = "+".join(
    [f"({XI[i]:.5f}*{feature_names[i[0]]})" for i in indices if i[1] == 1]
)
z2_dot_expr = "+".join(
    [f"({XI[i]:.5f}*{feature_names[i[0]]})" for i in indices if i[1] == 2]
)
z0_dot_expr = "z0_dot = " + z0_dot_expr if z0_dot_expr else "z0_dot = 0"
z1_dot_expr = "z1_dot = " + z1_dot_expr if z1_dot_expr else "z0_dot = 0"
z2_dot_expr = "z2_dot = " + z2_dot_expr if z2_dot_expr else "z0_dot = 0"
z0_dot_expr, z1_dot_expr, z2_dot_expr = [
    e.replace("sqrt", "np.sqrt")
    .replace("sign", "np.sign")
    .replace("|z", "np.abs(z")
    .replace("|)", "))")
    .replace(")np", ")*np")
    for e in [z0_dot_expr, z1_dot_expr, z2_dot_expr]
]
print(
    f"Identified model in latent space: \n{z0_dot_expr}\n{z1_dot_expr}\n{z2_dot_expr}"
)

# define ode model
def ode_model(z, t):
    z0 = z[0]
    z1 = z[1]
    z2 = z[2]

    # z0_dot = (0.00180*1)+(-0.01894*z0)+(0.01213*z1)+(-0.02310*z2)+(0.00075*z0*z1)+(0.00744*z0*z2)+(0.01403*z1*z2)+(-0.02050*z0*z0*z0)+(-0.00292*z0*z0*z1)+(-0.01276*z0*z0*z2)+(0.00613*z0*z1*z1)+(-0.01118*z0*z1*z2)+(-0.00058*z0*z2*z2)+(0.00765*z1*z1*z1)+(-0.00429*z1*z1*z2)+(-0.00259*z1*z2*z2)+(0.01472*z2*z2*z2)+(-0.01121*np.sign(z0)*np.sqrt(np.abs(z0)))+(0.00683*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.00089*np.sign(z2)*np.sqrt(np.abs(z2)))+(-0.02500*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(-0.00277*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(0.01002*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))
    # z1_dot = (-0.00315*1)+(0.00930*z0)+(-0.01261*z1)+(0.00704*z2)+(-0.00215*z0*z1)+(-0.00134*z0*z2)+(-0.00865*z1*z2)+(0.00669*z0*z0*z0)+(0.00575*z0*z0*z1)+(0.00105*z0*z0*z2)+(0.00323*z*z1*z1)+(-0.00589*z0*z1*z2)+(-0.00177*z0*z2*z2)+(-0.01404*z1*z1*z1)+(0.00497*z1*z1*z2)+(-0.01363*z1*z2*z2)+(0.00060*z2*z2*z2)+(0.00674*np.sign(z0)*np.sqrt(np.abs(z0)))+(-0.00627*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.00120*np.sign(z2)*np.sqrt(np.abs(z2)))+(0.02903*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(0.00381*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(-0.00484*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))
    # z2_dot = (-0.00259*1)+(-0.01811*z0)+(0.00085*z1)+(-0.04670*z2)+(0.00602*z0*z1)+(0.01858*z0*z2)+(0.02380*z1*z2)+(-0.01172*z0*z0*z0)+(-0.00567*z0*z0*z1)+(-0.00982*z0*z0*z2)+(0.00273*z0*z1*z1)+(-0.00151*z0*z1*z2)+(0.01738*z0*z2*z2)+(0.00240*z1*z1*z1)+(-0.01180*z1*z1*z2)+(0.01285*z1*z2*z2)+(-0.00932*z2*z2*z2)+(-0.00936*np.sign(z0)*np.sqrt(np.abs(z0)))+(0.00186*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.00225*np.sign(z2)*np.sqrt(np.abs(z2)))+(0.01378*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(0.00283*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(0.00849*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))

    z0_dot = (0.00215*1)+(-0.02024*z0)+(0.01363*z1)+(-0.02478*z2)+(0.00062*z0*z1)+(0.00688*z0*z2)+(0.01427*z1*z2)+(-0.02123*z0*z0*z0)+(-0.00290*z0*z0*z1)+(-0.01410*z0*z0*z2)+(0.00688*z0*z1*z1)+(-0.01127*z0*z1*z2)+(-0.00248*z0*z2*z2)+(0.00825*z1*z1*z1)+(-0.00410*z1*z1*z2)+(-0.00134*z1*z2*z2)+(0.01281*z2*z2*z2)+(-0.01135*np.sign(z0)*np.sqrt(np.abs(z0)))+(0.00757*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.02367*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(-0.00216*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(0.01012*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))
    z1_dot = (-0.00336*1)+(0.01074*z0)+(-0.01463*z1)+(0.00625*z2)+(-0.00209*z0*z1)+(-0.00065*z0*z2)+(-0.00953*z1*z2)+(0.00697*z0*z0*z0)+(0.00596*z0*z0*z1)+(0.00279*z0*z0*z2)+(0.00347*z0*z1*z1)+(-0.00572*z0*z1*z2)+(-0.00113*z0*z2*z2)+(-0.01496*z1*z1*z1)+(0.00542*z1*z1*z2)+(-0.01337*z1*z2*z2)+(0.00065*z2*z2*z2)+(0.00746*np.sign(z0)*np.sqrt(np.abs(z0)))+(-0.00734*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.00152*np.sign(z2)*np.sqrt(np.abs(z2)))+(0.02772*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(0.00339*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(-0.00380*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))
    z2_dot = (-0.00364*1)+(-0.01963*z0)+(0.00194*z1)+(-0.05201*z2)+(0.00591*z0*z1)+(0.02065*z0*z2)+(0.02591*z1*z2)+(-0.01200*z0*z0*z0)+(-0.00547*z0*z0*z1)+(-0.01024*z0*z0*z2)+(0.00316*z0*z1*z1)+(-0.00054*z0*z1*z2)+(0.01738*z0*z2*z2)+(0.00301*z1*z1*z1)+(-0.01177*z1*z1*z2)+(0.01277*z1*z2*z2)+(-0.00819*z2*z2*z2)+(-0.00925*np.sign(z0)*np.sqrt(np.abs(z0)))+(0.00310*np.sign(z1)*np.sqrt(np.abs(z1)))+(-0.00320*np.sign(z2)*np.sqrt(np.abs(z2)))+(0.01338*np.sign(z0-z1)*np.sqrt(np.abs(z0-z1)))+(0.00290*np.sign(z0-z2)*np.sqrt(np.abs(z0-z2)))+(0.00833*np.sign(z1-z2)*np.sqrt(np.abs(z1-z2)))

    return z0_dot, z1_dot, z2_dot



start_idx = 0
x_sample = x[start_idx, :]
z_sample = model.phi_x(x_sample.cuda())
t = np.linspace(0, 20, 50)
# t = np.array(range(1,51))
z_init = z_sample
z_pred = odeint(ode_model, z_sample.detach().cpu().numpy(), t).astype(np.float32)
x_pred = model.psi_z(torch.tensor(z_pred).cuda()).detach().cpu().numpy()


fig = make_subplots(rows=1, cols=1, shared_xaxes=True)
x_names = [r'$h_1(t)$', r'$h_2(t)$', r'$h_3(t)$']
xhat_names = [r'$\hat{h}_1(t)$', r'$\hat{h}_2(t)$', r'$\hat{h}_3(t)$']
DEFAULT_PLOTLY_COLORS=[
    'rgb(31, 119, 180)', 'rgb(255, 127, 14)',
                       'rgb(44, 160, 44)', 'rgb(214, 39, 40)',
                       'rgb(148, 103, 189)', 'rgb(140, 86, 75)',
                       'rgb(227, 119, 194)', 'rgb(127, 127, 127)',
                       'rgb(188, 189, 34)', 'rgb(23, 190, 207)']
for i, x_name, x_hat_name, colo in zip(range(3), x_names, xhat_names, DEFAULT_PLOTLY_COLORS):
    fig.add_trace(go.Scatter(x=list(range(50)), y=x[start_idx:start_idx+50, i+1],
                  mode="lines", opacity=1, name=x_name,
                  line = dict(shape = 'linear',
                              color = colo,
                              width = 2)), row=1, col=1)
    fig.add_trace(go.Scatter(x=list(range(50)), y=x_pred[:, i+1],
                  mode="lines", opacity=1, name=x_hat_name,
                  line = dict(shape = 'linear',
                              color = colo,
                              width = 2, dash = 'dot')), row=1, col=1)
    fig.update_xaxes(title_text='time', row=1)
    fig.update_yaxes(title_text='fill level', row=1)
    fig.update_layout(title_text="Predicted vs true fill levels", showlegend=True,
                     width=500, height=325, font_family='Serif', font_size=14)

pio.write_image(fig, const.FIGURE_PATH_RESULTS, width=500, height=325)
