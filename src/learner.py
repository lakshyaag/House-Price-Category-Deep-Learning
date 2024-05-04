from typing import List, Optional

import torch
from network import HousingNetwork
import lightning as L
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from torch import nn
from torch.nn import functional as F

torch.manual_seed(42)


class HousingModel(L.LightningModule):
    def __init__(
        self,
        model: HousingNetwork,
        eta,
        lr: float = 1e-3,
        class_weights: Optional[List[float]] = None,
    ):
        """
        Initializes the HousingModel which is a PyTorch Lightning module for housing data.

        Parameters:
            model (HousingNetwork): An instance of the HousingNetwork class configured with the necessary parameters.
            eta (List[float]): A list of scalars for the eta parameter in the combined loss function.
            lr (float): Learning rate for the optimizer. Defaults to 1e-3.
            class_weights (Optional[List[float]]): Class weights for handling imbalanced data in category prediction. Defaults to None.
        """
        super().__init__()

        self.model = model

        self.lr = lr

        self.loss_fn_price = nn.MSELoss()

        if class_weights is not None:
            class_weights = torch.tensor(class_weights, dtype=torch.float32)
            self.loss_fn_category = nn.CrossEntropyLoss(weight=class_weights)
        else:
            self.loss_fn_category = nn.CrossEntropyLoss()

        self.eta = nn.Parameter(torch.tensor(eta))

        hparams = {
            "lr": self.lr,
            "class_weights": class_weights,
            "eta": self.eta,
            "model__params": self.model.hparams,
            "model": self.model,
        }

        self.save_hyperparameters(hparams)

    def compute_losses(self, y_pred_price, y_price, y_pred_category, y_category):
        # Compute MSE for price
        mse_loss = self.loss_fn_price(y_pred_price.squeeze(), y_price)

        # Compute Cross-Entropy Loss
        nll_loss = self.loss_fn_category(y_pred_category, y_category.squeeze())

        combined_loss = (
            torch.stack([mse_loss, nll_loss]) * torch.exp(-self.eta) + self.eta
        ).sum()

        return combined_loss, mse_loss, nll_loss

    def forward(self, cat_features, num_features):
        return self.model(cat_features, num_features)

    def training_step(self, batch, batch_idx):
        cat_features, num_features, y_price, y_category = batch

        y_pred_price, y_pred_category = self(cat_features, num_features)

        loss, loss_price, loss_category = self.compute_losses(
            y_pred_price, y_price, y_pred_category, y_category
        )

        self.log_dict(
            {
                "train_loss": loss,
                "train_loss_price": loss_price,
                "train_loss_category": loss_category,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        self.log("eta1", self.eta[0], on_step=False, on_epoch=True, logger=True)
        self.log("eta2", self.eta[1], on_step=False, on_epoch=True, logger=True)

        self.log(
            "scaled_mse_loss",
            loss_price * torch.exp(-self.eta[0]) + self.eta[0],
            on_step=False,
            on_epoch=True,
            logger=True,
        )
        self.log(
            "scaled_nll_loss",
            loss_category * torch.exp(-self.eta[1]) + self.eta[1],
            on_step=False,
            on_epoch=True,
            logger=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        cat_features, num_features, y_price, y_category = batch

        y_pred_price, y_pred_category = self(cat_features, num_features)

        loss, loss_price, loss_category = self.compute_losses(
            y_pred_price, y_price, y_pred_category, y_category
        )

        self.log_dict(
            {
                "val_loss": loss,
                "val_loss_price": loss_price,
                "val_loss_category": loss_category,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        y_true = y_category.detach().cpu().numpy()
        y_pred = torch.argmax(y_pred_category, dim=-1).detach().cpu().numpy()

        accuracy_score_category = accuracy_score(y_true, y_pred)
        precision_score_category = precision_score(y_true, y_pred, average="weighted")
        recall_score_category = recall_score(y_true, y_pred, average="weighted")
        f1_score_category = f1_score(y_true, y_pred, average="weighted")

        self.log_dict(
            {
                "accuracy_score_category": accuracy_score_category,
                "precision_score_category": precision_score_category,
                "recall_score_category": recall_score_category,
                "f1_score_category": f1_score_category,
            },
            on_step=False,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

    def test_step(self, batch, batch_idx):
        cat_features, num_features, y_price, y_category = batch

        y_pred_price, y_pred_category = self(cat_features, num_features)

        loss, loss_price, loss_category = self.compute_losses(
            y_pred_price, y_price, y_pred_category, y_category
        )

        self.log_dict(
            {
                "test_loss": loss,
                "test_loss_price": loss_price,
                "test_loss_category": loss_category,
            },
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            logger=True,
        )

        return loss

    def predict_step(self, batch, batch_idx):
        (
            cat_features,
            num_features,
            _,
            _,
        ) = batch

        y_pred_price, y_pred_category = self(cat_features, num_features)

        # Apply softmax to each sample in the batch to ensure independent normalization
        y_pred_category = F.softmax(y_pred_category, dim=-1)

        return y_pred_price, y_pred_category.argmax(dim=-1)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.lr, weight_decay=1e-2)

        return optimizer
