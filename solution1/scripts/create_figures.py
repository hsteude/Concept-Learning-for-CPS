import numpy as np
import plotly.io as pio
from scipy.integrate import odeint
from plotly.subplots import make_subplots
import plotly.graph_objects as go
import solution1.constants as const
import os


##### For time series plot

# define constants
A = const.A
g = const.G
C = np.sqrt(2 * g) / A
q1 = [1, 2]
t = np.linspace(0, 10, 50)
q3 = [2, 1]
kv1 = [0.5, 1]
kv2 = [1, 0.5]


# define ODE model
def system_dynamics_function(x, t, q1, q3, kv1, kv2):
    x1 = x[0]
    x2 = x[1]
    x3 = x[2]
    dh1_dt = C * q1 - kv1 * C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))
    dh2_dt = kv1 * C * np.sign(x1 - x2) * np.sqrt(np.abs(x1 - x2))\
        - kv2 * C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
    dh3_dt = C * q3 + kv2 * C * np.sign(x2 - x3) * np.sqrt(np.abs(x2 - x3))
    return dh1_dt, dh2_dt, dh3_dt

# initial condition
x0 = (10, 100, 33)

# solve ode
y0 = odeint(system_dynamics_function, x0, t, (q1[0], q3[0], kv1[0], kv1[0]))
y1 = odeint(system_dynamics_function, x0, t, (q1[1], q3[1], kv1[1], kv1[1]))

# create figure
fig = make_subplots(
    rows=1, cols=2, shared_xaxes=True,
    subplot_titles=("Time series a", "Time series b")
)

# signal 1
DEFAULT_PLOTLY_COLORS = [
    "rgb(31, 119, 180)",
    "rgb(255, 127, 14)",
    "rgb(44, 160, 44)",
    "rgb(214, 39, 40)",
    "rgb(148, 103, 189)",
    "rgb(140, 86, 75)",
    "rgb(227, 119, 194)",
    "rgb(127, 127, 127)",
    "rgb(188, 189, 34)",
    "rgb(23, 190, 207)",
]
names0 = [r"$h_{1,a}(t)$", r"$h_{2,a}(t)$", r"$h_{3,a}(t)$"]
names1 = [r"$h_{1,b}(t)$", r"$h_{2,b}(t)$", r"$h_{3,b}(t)$"]
# names0 = [r'a', 'c', 'b']
# names1 = [r'a', 'c', 'b']
for color, sig, name in zip(
    DEFAULT_PLOTLY_COLORS[0:3], [y0[:, 0], y0[:, 1], y0[:, 2]], names0):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig,
            name=name,
            mode="lines",
            opacity=1),
        row=1,
        col=1,
    )
for color, sig, name in zip(
    DEFAULT_PLOTLY_COLORS[0:3], [y1[:, 0], y1[:, 1], y1[:, 2]], names1):
    fig.add_trace(
        go.Scatter(
            x=t,
            y=sig,
            name=name,
            mode="lines",
            opacity=1),
        row=1,
        col=2,
    )
fig.update_xaxes(title_text=r"time", title_font_family="Serif",
                 title_font_size=11)
fig.update_yaxes(
    title_text="fill level", row=1, col=1, title_font_family="Serif",
    title_font_size=11
)
fig.update_layout(
    width=500,
    height=225,
    font_family="Serif",
    font_size=14,
    legend_title_font_family="Serif",
    margin_l=5,
    margin_t=50,
    margin_b=5,
    margin_r=5,
)

pio.write_image(fig, const.FIGURE_PATH_TIME_SERIES, width=500, height=225)



#### For scatter matrix
from solution1.datagen.dataset import ThreeTankDataSet
from torch.utils.data import DataLoader
import yaml
import numpy as np
import pandas as pd
from plotly.subplots import make_subplots
import plotly.graph_objects as go
from solution1.seq2seq_vae.vae import VAE


dataset = ThreeTankDataSet()
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=24)
MODEL_VERSION = 'version_0'

hparams_path = f'./lightning_logs/{MODEL_VERSION}/hparams.yaml'
with open(hparams_path, 'r') as stream:
        hparam_dct = yaml.safe_load(stream)
ckpt_file_name = os.listdir(f'./lightning_logs/{MODEL_VERSION}/checkpoints/')[-1]
ckpt_file_path = f'./lightning_logs/{MODEL_VERSION}/checkpoints/{ckpt_file_name}'
model = VAE.load_from_checkpoint(ckpt_file_path)
model


batches = iter(dataloader)
x_batch, labels_batch, idxs_batch = batches.next()
mu_z, std_z, z_sample, mu_x, std_x = model.eval()(x_batch)

df_latent_mu = pd.DataFrame(mu_z.detach().numpy(),
                            columns=[f'mu_{i}' for i in range(hparam_dct['latent_dim'])])

df_real_params = pd.DataFrame(labels_batch.numpy(), columns=const.LABEL_COLS)
df_real_params['sample_idx'] = idxs_batch
df_real_params.head()

fig = make_subplots(rows=4, cols=5)

for i, hs in enumerate(const.LABEL_COLS):
    for j, hs_pred in enumerate(df_latent_mu.columns):
        fig.add_trace(go.Scatter(y=df_latent_mu[hs_pred], x=df_real_params[hs], 
                            mode='markers', name=f'activation {hs_pred} over box_x',
                                marker_color='#1f77b4'),
                     row=i+1, col=j+1)
        fig.update_yaxes(range=[-5, 5])

# Update xaxis properties
for i in range(hparam_dct['latent_dim']):
    fig.update_xaxes(title_text=df_real_params.columns[0], row=1, col=i+1)
    fig.update_xaxes(title_text=df_real_params.columns[1], row=2, col=i+1)
    fig.update_xaxes(title_text=df_real_params.columns[2], row=3, col=i+1)
    fig.update_xaxes(title_text=df_real_params.columns[3], row=4, col=i+1)

# Update xaxis properties
for j in range(len(df_real_params)):
    fig.update_yaxes(title_text=df_latent_mu.columns[0], row=j+1, col=1)
    fig.update_yaxes(title_text=df_latent_mu.columns[1], row=j+1, col=2)
    fig.update_yaxes(title_text=df_latent_mu.columns[2], row=j+1, col=3)
    fig.update_yaxes(title_text=df_latent_mu.columns[3], row=j+1, col=4)
    fig.update_yaxes(title_text=df_latent_mu.columns[4], row=j+1, col=5)


fig.update_layout(height=1000, width=1500, title_text="Latent neuron activations vs. hidden states", showlegend=False)

