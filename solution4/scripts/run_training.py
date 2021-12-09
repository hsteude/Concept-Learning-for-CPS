import glob
import pytorch_lightning as pl
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import ModelCheckpoint, LearningRateMonitor, EarlyStopping

import solution4.constants as const
from solution4.datagen.datamodule import ThreeTankStateDataModule
from solution4.som_vae.som_vae import SOMVAE


def train():
    pl.seed_everything(1234)
    dm = ThreeTankStateDataModule(
        nb_of_samples=10000,
        window_size=100,
        ordered_samples=False,
        batch_size=const.BATCH_SIZE,
        num_workers=const.NUM_WORKERS,
        pin_memory=False)

    # trainer callbacks
    ckpt_callback = ModelCheckpoint(filename='{epoch}-{val_loss:.2f}')
    lr_monitor = LearningRateMonitor()
    early_stop = EarlyStopping(monitor="val_loss", patience=15)
    if const.USE_LOGGER:
        logger = TensorBoardLogger(const.LOGDIR, name=const.MODEL_NAME, default_hp_metric=False)
    else:
        logger = False

    trainer = pl.Trainer(**const.TRAINER_CONFIG,
                         callbacks=[ckpt_callback, lr_monitor, early_stop],
                         logger=logger)

    # train
    model = SOMVAE(**const.HPARAMS)
    trainer.fit(model=model, datamodule=dm)
    trainer.test(model=model, datamodule=dm)


if __name__ == '__main__':
    train()
