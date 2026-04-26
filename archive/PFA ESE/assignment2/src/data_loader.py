"""
data_loader.py
--------------
Loads the Heart Disease dataset from CSV file.
Separates features (X) and target variable (y).
"""

import pandas as pd
from config import DATA_PATH


def load_data(filepath=None):
    """
    Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to DATA_PATH from config.py.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except 'target').
    y : pd.Series
        Target variable (heart disease risk: 0 = low risk, 1 = high risk).
    """
    if filepath is None:
        filepath = DATA_PATH

    df = pd.read_csv(filepath, encoding='utf-8-sig')

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"[DataLoader] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y
