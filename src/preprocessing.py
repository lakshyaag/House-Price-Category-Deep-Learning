from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sklearn.compose import make_column_selector
import numpy as np
import pandas as pd
from typing import Tuple, List, Dict


def split_data(
    train: pd.DataFrame
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series, pd.Series]:
    """
    Splits the dataset into training and validation sets, separating features from target variables.

    Parameters:
    - train (pd.DataFrame): The input dataset containing features and target variables.

    Returns:
    - Tuple containing training and validation sets for features and target variables (prices and categories).
    """
    X = train.drop(["Id", "SalePrice", "HouseCategory"], axis=1)
    y_price = train["SalePrice"]
    y_category = train["HouseCategory"]

    (
        X_train,
        X_val,
        y_train_category,
        y_val_category,
    ) = train_test_split(
        X,
        y_category,
        test_size=0.3,
        random_state=42,
        stratify=y_category,
    )

    y_train_price, y_val_price = (
        y_price.iloc[y_train_category.index],
        y_price.iloc[y_val_category.index],
    )

    return X_train, X_val, y_train_price, y_val_price, y_train_category, y_val_category


features_na_none = [
    "PoolQC",
    "MiscFeature",
    "Alley",
    "Fence",
    "MasVnrType",
    "FireplaceQu",
    "GarageType",
    "GarageFinish",
    "GarageQual",
    "GarageCond",
    "BsmtQual",
    "BsmtCond",
    "BsmtExposure",
    "BsmtFinType1",
    "BsmtFinType2",
]

features_na_median = ["LotFrontage"]

features_na_zero = ["GarageYrBlt", "GarageArea", "GarageCars", "MasVnrArea"]

numerical_to_categorical = [
    "MSSubClass",
    "OverallCond",
    "OverallQual",
    "YrSold",
    "MoSold",
]

num_to_cat = FunctionTransformer(
    lambda x: x.astype(str), feature_names_out="one-to-one"
)

imputer_none = SimpleImputer(strategy="constant", fill_value="None")
imputer_median = SimpleImputer(strategy="median")
imputer_zero = SimpleImputer(strategy="constant", fill_value=0)

preprocessor = ColumnTransformer(
    transformers=[
        ("imputer_none", imputer_none, features_na_none),
        ("imputer_median", imputer_median, features_na_median),
        ("imputer_zero", imputer_zero, features_na_zero),
        ("num_to_cat", num_to_cat, numerical_to_categorical),
    ],
    remainder="passthrough",
)

full_pipeline = Pipeline(steps=[("preprocessor", preprocessor)])

scaler_pipeline = Pipeline(
    [
        (
            "scaler",
            ColumnTransformer(
                transformers=[
                    (
                        "scaler",
                        StandardScaler(),
                        make_column_selector(dtype_include="number"),
                    ),
                ],
                remainder="passthrough",
            ),
        )
    ]
)


def run_pipeline(X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
    """
    Applies preprocessing and scaling pipelines to the input data.

    Parameters:
    - X (pd.DataFrame): The input data to be processed.
    - fit (bool): Indicates whether the pipeline should be fitted to the data. Default is False.

    Returns:
    - pd.DataFrame: The processed and scaled data.
    """
    if fit:
        transformed_data = full_pipeline.fit_transform(X)
    else:
        transformed_data = full_pipeline.transform(X)

    x = pd.DataFrame(
        transformed_data, columns=full_pipeline.get_feature_names_out()
    ).infer_objects()

    if fit:
        scaled_data = scaler_pipeline.fit_transform(x)
    else:
        scaled_data = scaler_pipeline.transform(x)

    x = pd.DataFrame(
        scaled_data, columns=scaler_pipeline.get_feature_names_out()
    ).infer_objects()

    x.columns = [y.split("__")[-1] for y in x.columns]

    return x


def get_column_by_type(X: pd.DataFrame) -> Tuple[List[str], List[str]]:
    """
    Identifies numerical and categorical columns in the input data.

    Parameters:
    - X (pd.DataFrame): The input data.

    Returns:
    - Tuple[List[str], List[str]]: Lists of numerical and categorical column names.
    """
    num_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    cat_cols = X.select_dtypes(include=["object"]).columns.tolist()

    return num_cols, cat_cols


def label_encode(
    X_train: pd.DataFrame,
    X_val: pd.DataFrame,
    X_test: pd.DataFrame,
    cat_cols: List[str],
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, Dict[str, LabelEncoder]]:
    """
    Applies label encoding to categorical columns.

    Parameters:
    - X_train (pd.DataFrame): The training data.
    - X_val (pd.DataFrame): The validation data.
    - X_test (pd.DataFrame): The test data.
    - cat_cols (List[str]): List of categorical column names to be encoded.

    Returns:
    - Tuple containing the transformed training, validation, and test data, along with the label encoders used.
    """
    label_encoders = {}

    for col in cat_cols:
        le = LabelEncoder()
        X_train[col] = le.fit_transform(X_train[col])
        label_encoders[col] = le

    for col in cat_cols:
        le = label_encoders[col]
        le.classes_ = np.append(le.classes_, "Unknown")

        X_val[col] = X_val[col].map(lambda s: "Unknown" if s not in le.classes_ else s)
        X_val[col] = le.transform(X_val[col])

        X_test[col] = X_test[col].map(
            lambda s: "Unknown" if s not in le.classes_ else s
        )
        X_test[col] = le.transform(X_test[col])

    return X_train, X_val, X_test, label_encoders


def categorize_house(row: pd.Series) -> str:
    """
    Categorizes a house based on its year built, style, and building type.

    Parameters:
    - row (pd.Series): A row of data representing a house.

    Returns:
    - str: The category of the house.
    """
    if row["YearBuilt"] != row["YearRemodAdd"]:
        age_category = "Remodeled"
    else:
        year_diff = 2000 - row["YearBuilt"]
        if year_diff <= 15:
            age_category = "Recent"
        elif year_diff <= 30:
            age_category = "Modern"
        else:
            age_category = "Historic"

    house_style = (
        "Single"
        if row["HouseStyle"] in {"1Story", "1.5Fin", "1.5Unf"}
        else "Multi-Storey"
    )
    bldg_type = "1Fam" if row["BldgType"] == "1Fam" else "Townhouse/Duplex"

    return f"{age_category}-{house_style}-{bldg_type}"
