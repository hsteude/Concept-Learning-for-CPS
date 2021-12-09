import os
import glob
import numpy as np
import pandas as pd
import torch
import yaml

from plotly.subplots import make_subplots
import plotly.graph_objects as go
import plotly.io as pio

import solution4.constants as const
from solution4.som_vae.som_vae import SOMVAE


def plot_simulation(y: np.array):
    """Plots the tank levels of the three tank system"""
    fig = make_subplots(rows=1, cols=1, shared_xaxes=True)

    for sig, name in zip([y[:, 0], y[:, 1], y[:, 2]],
                         ['h1', 'h2', 'h3']):
        fig.add_trace(go.Scatter(x=np.array(range(y.shape[0])), y=sig, name=name,
                      mode="lines", opacity=1),
                      row=1, col=1)

    fig.update_xaxes(title_text=r'time', row=2, col=1)
    fig.update_yaxes(title_text='fill level')
    fig.update_layout(width=500, height=400,
                      font_family="Serif", font_size=14,
                      margin_l=5, margin_t=50, margin_b=5, margin_r=5
                      )
    fig.show()


def plot_state_prediction(model, start_idx=None, stop_idx=None, step=1, title=None, width=500, height=400):
    """Plots section of datagen with a prediction of the underlying state."""
    # load datagen
    df = pd.read_csv("../../data/solution_4_dataset.csv")

    # default: plot section of test datagen
    if start_idx is None:
        start_idx = 1200
    if stop_idx is None:
        stop_idx = 2550
    df = df[start_idx:stop_idx]


    # get state prediction
    states = pd.DataFrame(index=df.index)
    for i in range(start_idx, stop_idx - 100, step):
        x = df.loc[i:i+99, :].to_numpy(dtype=np.float32)
        x = torch.from_numpy(x)
        x = x.unsqueeze(0)
        states.loc[i+100, 0] = model(x).squeeze().numpy()

    df = df.iloc[100:, :]
    states = states.iloc[100:, :]
    states.dropna(inplace=True)

    # plot
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True,
                        vertical_spacing=0.05)

    for sig, name in zip([df.iloc[:, 0], df.iloc[:, 1], df.iloc[:, 2]],
                         [r'$h_1(t)$', r'$h_2(t)$', r'$h_3(t)$']):
        fig.add_trace(go.Scatter(x=df.index, y=sig, name=name,
                                 mode="lines", opacity=1),
                      row=1, col=1)

    fig.add_trace(go.Scatter(x=states.index, y=states.values.squeeze(), name=r"$\text{state}(t)$",
                             mode="lines", opacity=1),
                  row=2, col=1)

    fig.update_xaxes(title_text=r'time', row=2, col=1)
    fig.update_yaxes(title_text='fill level', row=1, col=1)
    fig.update_yaxes(title_text='state', row=2, col=1)
    fig.update_layout(width=width, height=height,
                      font_family="Serif", font_size=14,
                      margin_l=5, margin_t=50, margin_b=5, margin_r=5)
    if title is not None:
        fig.update_layout(title=title)

    pio.write_image(fig, f"../visualizations/figures/state-plot.pdf", width=width, height=height)
    fig.show()


def _get_ckpt_path(logdir, model_name, version=0):
    path = f"{logdir}/{model_name}/version_{version}/checkpoints/*.ckpt"
    return glob.glob(path)[0]


def _get_hparams(logdir, model_name, version=0):
    path = f"{logdir}/{model_name}/version_{version}/hparams.yaml"
    with open(path, "r") as file:
        hparams = yaml.safe_load(file)
    return hparams


def load_model(logdir, model_name, version=0):
    ckpt_path = _get_ckpt_path(logdir, model_name, version)
    hparams = _get_hparams(logdir, model_name, version)
    return SOMVAE.load_from_checkpoint(ckpt_path, **hparams)


if __name__ == '__main__':
    plot_state_prediction(load_model(const.LOGDIR, const.MODEL_NAME),
                          title="Fill levels with predicted state")
