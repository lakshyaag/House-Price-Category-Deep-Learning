from dataclasses import dataclass

from typing import List
import torch.nn as nn

@dataclass
class NetworkParameters:
    """
    A dataclass to hold the parameters for the HousingNetwork model.

    Parameters:
        cat_cols (List[str]): List of names for categorical columns.
        num_cols (List[str]): List of names for numerical columns.
        num_layers (int): Number of additional linear layers in the network.
        num_embds (List[int]): List specifying the number of embeddings for each categorical feature.
        embedding_dim (int): Dimensionality of the embeddings for categorical features.
        hidden_dim (int): Dimensionality of the hidden layers.
        output_dim_price (int): Dimensionality of the output layer for price prediction.
        output_dim_category (int): Dimensionality of the output layer for category classification.
        activation (nn.Module): Activation function to use in the network. Defaults to nn.LeakyReLU().
    """
    cat_cols: List[str]
    num_cols: List[str]
    num_layers: int
    num_embds: List[int]
    embedding_dim: int
    hidden_dim: int
    output_dim_price: int
    output_dim_category: int
    activation: nn.Module
