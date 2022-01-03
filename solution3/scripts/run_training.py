from solution3.datagen.data_module import ThreeTankDataModule
from pytorch_lightning.loggers import TensorBoardLogger
from solution3.ae_sindy.ae_sindy import SINDyAutoencoder
import torch
import pytorch_lightning as pl
from solution3.ae_sindy.callbacks import SequentialThresholdingCallback
import solution3.constants as const


HPARAMS = dict(
    learning_rate=1e-4,
    input_dim=56, 
    latent_dim=3,
    # enc_hidden_sizes=[512, 128, 64, 32],
    # dec_hidden_sizes=[32, 64, 128, 512],
    enc_hidden_sizes=[1024, 512, 128, 64],
    dec_hidden_sizes=[64, 128, 512, 1024],
    activation='tanh',
    validdation_split=.1,
    batch_size=32,
    dl_num_workers=8,
    sindy_biases=True,
    sindy_states=True,
    sindy_sin=False,
    sindy_cos=False,
    sindy_multiply_pairs=True,
    sindy_poly_order=2,
    sindy_sqrt=True,
    sindy_inverse=False,
    sindy_sign_sqrt_of_diff=True,
    sequential_thresholding=True,
    sequential_thresholding_freq = 500,
    sequential_thresholding_thres = 0.0005,
    loss_weight_sindy_x=5e3,
    loss_weight_sindy_regularization=1e-6,
    loss_weight_sindy_z=1e2,
    max_epochs=5010,
)

def train():
    gpus = 1 if torch.cuda.is_available() else 0
    logger = TensorBoardLogger(const.LOGDIR, name=const.MODEL_NAME,
                               default_hp_metric=True)
    trainer = pl.Trainer(
        gradient_clip_val=0.1,
        gpus=gpus,
        max_epochs=HPARAMS["max_epochs"],
        precision=64,
        callbacks=[SequentialThresholdingCallback()],
    logger=logger)
    model = SINDyAutoencoder(**HPARAMS)
    dm = ThreeTankDataModule(validdation_split=HPARAMS['validdation_split'],
                             batch_size=HPARAMS['batch_size'],
                             dl_num_workers=HPARAMS['dl_num_workers'])
    trainer.fit(model, dm)


if __name__ == '__main__':
    train()
