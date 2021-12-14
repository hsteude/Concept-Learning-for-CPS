from solution2.datagen.dataset import ThreeTankDataSet
import plotly.io as pio
import solution2.constants as const
from torch.utils.data import DataLoader
from solution2.comm_agents.lightning_module import LitModule
import yaml
import os
from plotly.subplots import make_subplots
import pandas as pd
import plotly.graph_objects as go


dataset = ThreeTankDataSet()
batch_size = 500
dataloader = DataLoader(dataset, batch_size=batch_size, num_workers=24)


# MODEL_VERSION = 'freq_and_phase'
MODEL_VERSION = 'version_1'
hparams_path = f'./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/hparams.yaml'
with open(hparams_path, 'r') as stream:
        hparam_dct = yaml.safe_load(stream)
ckpt_file_name = os.listdir(f'./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/checkpoints/')[-1]
ckpt_file_path = f'./{const.LOGDIR}/{const.MODEL_NAME}/{MODEL_VERSION}/checkpoints/{ckpt_file_name}'
model = LitModule.load_from_checkpoint(ckpt_file_path)


batches = iter(dataloader)
x, answers, labels, idxs = batches.next()
z, ans_pred = model.cuda()(x.cuda())


df_latent = pd.DataFrame(z.cpu().detach().numpy(),
                            columns=[f'z{i}' for i in range(5)])
df_latent.head()


df_real_params = pd.DataFrame(labels.numpy(), columns=const.LABEL_COLS)
df_real_params['sample_idx'] = idxs
df_real_params.head()


fig = make_subplots(rows=4, cols=5)

for i, hs in enumerate(const.LABEL_COLS):
    for j, hs_pred in enumerate(df_latent.columns):
        fig.add_trace(go.Scatter(y=df_latent[hs_pred], x=df_real_params[hs], 
                            mode='markers', name=f'activation {hs_pred} over box_x',
                                marker_color='#1f77b4',
                                marker=dict(size=3),
),
                     row=i+1, col=j+1)


fig.update_layout(title_text=r"Latent space activation over true concepts",
                  showlegend=False,
                  width =500, height=300, 
                  font_family="Serif", font_size=11, 
                  margin_l=5, margin_t=50, margin_b=5, margin_r=5,
)


pio.write_image(fig, const.FIGURE_PATH_RESULTS, width=500, height=300)
