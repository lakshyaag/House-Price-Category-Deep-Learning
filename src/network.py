import torch
import torch.nn as nn

from classes import NetworkParameters

torch.manual_seed(42)


class HousingNetwork(nn.Module):
    def __init__(
        self,
        params: NetworkParameters,
    ):
        """
        Initializes the HousingNetwork model with specific configurations for handling housing data.

        Parameters:
            params (NetworkParameters): An instance of the NetworkParameters class containing the necessary parameters for the model.
        """
        super().__init__()

        self.hparams = params

        self.cat_cols = params.cat_cols
        self.num_cols = params.num_cols
        self.num_layers = params.num_layers
        self.num_embds = params.num_embds
        self.embedding_dim = params.embedding_dim
        self.hidden_dim = params.hidden_dim
        self.output_dim_price = params.output_dim_price
        self.output_dim_category = params.output_dim_category
        self.total_embeddings = len(self.cat_cols) * self.embedding_dim
        self.activation = params.activation()

        # Create embedding layers for categorical features
        self.embeddings = nn.ModuleList(
            [
                nn.Embedding(self.num_embds[i], self.embedding_dim)
                for i in range(len(self.cat_cols))
            ]
        )

        # Create first shared linear layer for numerical and embedded categorical features
        self.fc = nn.Sequential(
            nn.Linear(len(self.num_cols) + self.total_embeddings, self.hidden_dim * 2),
            self.activation,
            nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
            self.activation,
        )

        # Create additional linear layers as specified by num_layers
        self.shared_layers = nn.ModuleList()
        for _ in range(self.num_layers):
            self.shared_layers.append(
                nn.Sequential(
                    nn.Linear(self.hidden_dim * 2, self.hidden_dim * 2),
                    nn.BatchNorm1d(self.hidden_dim * 2),
                    self.activation,
                )
            )

        # Create output layers for each task
        self.output_price = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.Linear(self.hidden_dim, self.output_dim_price),
        )

        self.output_category = nn.Sequential(
            nn.Linear(self.hidden_dim * 2, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim_category),
        )

        # Kaiming initialization for all layers
        for emb in self.embeddings:
            nn.init.kaiming_uniform_(emb.weight.data)

        for layer in self.fc:
            if isinstance(layer, nn.Linear):
                nn.init.kaiming_uniform_(layer.weight.data)

        for mod in [self.output_price, self.output_category]:
            for layer in mod:
                if isinstance(layer, nn.Linear):
                    nn.init.kaiming_uniform_(layer.weight.data)

    def forward(
        self,
        cat_features: torch.Tensor,
        num_features: torch.Tensor,
    ) -> torch.Tensor:
        """
        Forward pass of the network, processing inputs through the model to produce outputs

        Args:
            cat_features (torch.Tensor): The tensor containing the categorical features.
            num_features (torch.Tensor): The tensor containing the numerical features.

        Returns:
            torch.Tensor: The output tensor from the model, with separate outputs for price and category predictions.
        """

        # Pass categorical features through embedding layers
        embedded_features = []
        for i, emb in enumerate(self.embeddings):
            embedded_features.append(emb(cat_features[:, i]))

        embedded_features = torch.cat(embedded_features, dim=1)

        # Concatenate embedded categorical features and numerical features
        concat_features = torch.cat([embedded_features, num_features], dim=1)
        concat_features = self.fc(concat_features)

        # Pass through additional layers
        for layer in self.shared_layers:
            concat_features = layer(concat_features)

        return [
            self.output_price(concat_features),
            self.output_category(concat_features),
        ]
