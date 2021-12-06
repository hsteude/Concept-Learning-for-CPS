from solution1.seq2seq_vae.vae import VAE
from pytorch_lightning.loggers import TensorBoardLogger
from solution1.datagen.data_module import ThreeTankDataModule
import torch
import pytorch_lightning as pl
import solution1.constants as const


HPARAMS = dict(
        # datamodule
        validation_split=.1,
        test_split=.05,
        batch_size=100,
        dl_num_workers=6,
        seq_len=50,
        max_epochs=2437,
        learning_rate=1e-3,
        latent_dim=5,
        input_size=3,
        beta=1,
        enc_out_dim=100,
        hidden_size=100,
        rnn_layers=5,
        rnn_models=True,
    )


def train():
    gpus = 1 if torch.cuda.is_available() else 0
    logger = TensorBoardLogger(const.LOGDIR, name=const.MODEL_NAME,
                               default_hp_metric=False)
    trainer = pl.Trainer(
        gpus=gpus,
        max_epochs=HPARAMS['max_epochs'],
        logger=logger)
    vae = VAE(**HPARAMS)
    dm = ThreeTankDataModule(**HPARAMS)
    trainer.fit(vae, dm)


if __name__ == '__main__':
    train()
