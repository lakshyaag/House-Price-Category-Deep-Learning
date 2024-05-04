from typing import List, Tuple

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

torch.manual_seed(42)


class HousingDataset(Dataset):
    """
    A custom dataset for housing data that handles both categorical and numerical features.

    Attributes:
        X (pd.DataFrame): The input features dataframe.
        y_price (np.ndarray): The target prices array.
        y_category (np.ndarray): The target categories array.
        cat_cols (list): List of column names for categorical features.
        num_cols (list): List of column names for numerical features.
    """

    def __init__(
        self,
        X: pd.DataFrame,
        y_price: np.ndarray,
        y_category: np.ndarray,
        cat_cols: List[str],
        num_cols: List[str],
    ):
        """
        Initializes the HousingDataset with data and feature information.

        Args:
            X (pd.DataFrame): The input features dataframe.
            y_price (np.ndarray): The target prices array.
            y_category (np.ndarray): The target categories array.
            cat_cols (list): List of column names for categorical features.
            num_cols (list): List of column names for numerical features.
        """
        self.X = X
        self.y_price = y_price
        self.y_category = y_category
        self.cat_cols = cat_cols
        self.num_cols = num_cols

    def __len__(self) -> int:
        """
        Returns the total number of samples in the dataset.

        Returns:
            int: The total number of samples.
        """
        return len(self.X)

    def __getitem__(
        self, idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Retrieves the features and labels at the specified index.

        Args:
            idx (int): The index of the data point to retrieve.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing categorical features,
            numerical features, price label, and category label, all as tensors.
        """
        x = self.X.iloc[idx]
        y_price = self.y_price[idx]
        y_category = self.y_category[idx]

        cat_features = torch.tensor(x[self.cat_cols].values, dtype=torch.long)
        num_features = torch.from_numpy(x[self.num_cols].values.astype(np.float32))

        return (
            cat_features,
            num_features,
            torch.tensor(y_price, dtype=torch.float32),
            torch.tensor(y_category, dtype=torch.long),
        )
