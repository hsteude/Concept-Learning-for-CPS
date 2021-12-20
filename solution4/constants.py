LOGDIR = "../../logs"
MODEL_NAME = "SOMVAE"
MODEL_VERSION = 0
BATCH_SIZE = 32
NUM_WORKERS = 8
USE_LOGGER = True

TRAINER_CONFIG = dict(
    gpus=1,
    max_epochs=150
)

HPARAMS = dict(
    d_som=[2, 3],
    d_latent=64,
    d_enc_dec=100,
    alpha=0.75,
    beta=2,
    # gamma=0,
    # tau=0
)
