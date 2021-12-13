from pytorch_lightning.callbacks import Callback
import torch


class SequentialThresholdingCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        its_time = trainer.current_epoch % pl_module.sequential_thresholding_freq == 0
        its_not_epoch_1 = trainer.current_epoch > 0
        if its_time and its_not_epoch_1 and pl_module.sequential_thresholding:
            pl_module.XI_coefficient_mask = (
                torch.abs(pl_module.XI) > pl_module.sequential_thresholding_thres
            )
            print(f"mask after: {pl_module.XI_coefficient_mask}")


class SequentialLossWeightsCallback(Callback):
    def on_train_epoch_end(self, trainer, pl_module):
        if trainer.current_epoch == 5:
            pl_module.loss_weight_sindy_x = 10
            print(f"Setting sindy loss x to {pl_module.loss_weight_sindy_x}")

        if trainer.current_epoch == 10:
            pl_module.loss_weight_sindy_z = 10
            print(f"Setting sindy loss z to {pl_module.loss_weight_sindy_x}")

        if trainer.current_epoch == 15:
            pl_module.loss_weight_sindy_regularization = 1e-4
            print(
                f"Setting XI regularization loss to {pl_module.loss_weight_sindy_regularization}"
            )
