import lightning as L
import numpy as np
import optuna
import torch
import torch.nn as nn
from classes import NetworkParameters
from dataset import HousingDataset
from learner import HousingModel
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.callbacks import EarlyStopping
from loss_callback import LossHistory
from network import HousingNetwork
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader


class PatchedCallback(optuna.integration.PyTorchLightningPruningCallback, L.Callback):
    pass


seed = torch.manual_seed(42)

class Objective(object):
    def __init__(
        self,
        X_train,
        X_val,
        y_train_price,
        y_val_price,
        y_train_category,
        y_val_category,
        cat_cols,
        num_cols,
        label_encoders,
        house_category_encoder,
        max_epochs,
    ):
        self.X_train = X_train
        self.X_val = X_val
        self.y_train_price = y_train_price
        self.y_val_price = y_val_price
        self.y_train_category = y_train_category
        self.y_val_category = y_val_category
        self.cat_cols = cat_cols
        self.num_cols = num_cols
        self.label_encoders = label_encoders
        self.house_category_encoder = house_category_encoder
        self.max_epochs = max_epochs

    def __call__(self, trial):
        train_loader = DataLoader(
            HousingDataset(
                self.X_train,
                self.y_train_price,
                self.y_train_category,
                self.cat_cols,
                self.num_cols,
            ),
            batch_size=32,
            shuffle=True,
            generator=seed,
        )

        val_loader = DataLoader(
            HousingDataset(
                self.X_val,
                self.y_val_price,
                self.y_val_category,
                self.cat_cols,
                self.num_cols,
            ),
            batch_size=32,
            shuffle=False,
            generator=seed,
        )

        logger = TensorBoardLogger("lightning_logs", name="housing-final-optuna")

        num_layers = trial.suggest_int("num_layers", 2, 6)
        embedding_dim = trial.suggest_categorical(
            "embedding_dim", [2**x for x in range(4, 9)]
        )
        hidden_dim = trial.suggest_categorical(
            "hidden_dim", [2**x for x in range(5, 10)]
        )
        
        lr = trial.suggest_loguniform("lr", 1e-5, 1e-2)

        params = NetworkParameters(
            cat_cols=self.cat_cols,
            num_cols=self.num_cols,
            num_layers=num_layers,
            num_embds=[len(le.classes_) for le in self.label_encoders.values()],
            embedding_dim=embedding_dim,
            hidden_dim=hidden_dim,
            output_dim_price=1,
            output_dim_category=len(self.house_category_encoder.classes_),
            activation=nn.LeakyReLU,
        )

        network = HousingNetwork(params)

        model = HousingModel(
            model=network,
            eta=[20.0, 0.0],
            lr=lr,
            class_weights=compute_class_weight(
                "balanced",
                classes=np.unique(self.y_train_category),
                y=self.y_train_category.squeeze(),
            ),
        )

        loss_history = LossHistory(params=params)

        trainer = L.Trainer(
            max_epochs=self.max_epochs,
            accelerator="gpu",
            check_val_every_n_epoch=1,
            log_every_n_steps=10,
            callbacks=[
                EarlyStopping(
                    monitor="val_loss_price",
                    patience=10,
                    mode="min",
                    min_delta=1e-4,
                    verbose=True,
                ),

                loss_history,
                PatchedCallback(trial=trial, monitor="val_loss"),
            ],
            logger=logger,
        )

        trainer.fit(model, train_loader, val_loader)

        return trainer.callback_metrics[
            "val_loss"
        ].item()
