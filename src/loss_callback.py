import numpy as np
import matplotlib.pyplot as plt
import lightning as L

from classes import NetworkParameters


class LossHistory(L.Callback):
    """
    A PyTorch Lightning callback to track and plot the loss curves during training and validation.

    Attributes:
        losses (dict): A dictionary to store the training and validation losses for price, category, and overall.
        params (NetworkParameters): The network parameters used for training.
    """

    def __init__(self, params: NetworkParameters):
        super().__init__()

        self.losses = {
            "train": {"price": [], "category": [], "overall": []},
            "val": {"price": [], "category": [], "overall": []},
        }

        self.params = params

    def on_train_epoch_end(self, trainer, pl_module, outputs=None):
        if "train_loss_price" in trainer.callback_metrics:
            self.losses["train"]["price"].append(
                np.sqrt(trainer.callback_metrics["train_loss_price"].item())
            )
        if "train_loss_category" in trainer.callback_metrics:
            self.losses["train"]["category"].append(
                trainer.callback_metrics["train_loss_category"].item()
            )
        if "train_loss" in trainer.callback_metrics:
            self.losses["train"]["overall"].append(
                trainer.callback_metrics["train_loss"].item()
            )

    def on_validation_epoch_end(self, trainer, pl_module):
        if "val_loss_price" in trainer.callback_metrics:
            self.losses["val"]["price"].append(
                np.sqrt(trainer.callback_metrics["val_loss_price"].item())
            )
        if "val_loss_category" in trainer.callback_metrics:
            self.losses["val"]["category"].append(
                trainer.callback_metrics["val_loss_category"].item()
            )
        if "val_loss" in trainer.callback_metrics:
            self.losses["val"]["overall"].append(
                trainer.callback_metrics["val_loss"].item()
            )

    def plot_loss_curves(self):
        fig, axs = plt.subplots(3, 1, figsize=(10, 18))

        # Plotting Price Loss
        axs[0].plot(
            self.losses["train"]["price"], label="Training Price Loss", color="blue"
        )
        axs[0].plot(
            self.losses["val"]["price"], label="Validation Price Loss", color="red"
        )
        axs[0].set_title("Price Loss Curves")
        axs[0].set_xlabel("Epochs")
        axs[0].set_ylabel("RMSE")
        axs[0].legend()

        # Plotting Category Loss
        axs[1].plot(
            self.losses["train"]["category"],
            label="Training Category Loss",
            color="blue",
        )
        axs[1].plot(
            self.losses["val"]["category"],
            label="Validation Category Loss",
            color="red",
        )
        axs[1].set_title("Category Loss Curves")
        axs[1].set_xlabel("Epochs")
        axs[1].set_ylabel("NLL")
        axs[1].legend()

        # Plotting Overall Loss
        axs[2].plot(
            self.losses["train"]["overall"], label="Training Overall Loss", color="blue"
        )
        axs[2].plot(
            self.losses["val"]["overall"], label="Validation Overall Loss", color="red"
        )
        axs[2].set_title("Overall Loss Curves")
        axs[2].set_xlabel("Epochs")
        axs[2].set_ylabel("Loss")
        axs[2].legend()
        plt.tight_layout()
        plt.savefig("./graphs/loss_curves.jpg", dpi=300)
        plt.show()
