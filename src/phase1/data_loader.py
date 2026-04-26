"""
data_loader.py
--------------
Loads the Heart Disease dataset from CSV file.
Separates features (X) and target variable (y).
"""

import pandas as pd
import os


def load_data(filepath=None):
    """
    Load the heart disease dataset from a CSV file.

    Parameters
    ----------
    filepath : str, optional
        Path to the CSV file. Defaults to 'data/heart.csv' relative to project root.

    Returns
    -------
    X : pd.DataFrame
        Feature matrix (all columns except 'target').
    y : pd.Series
        Target variable (heart disease risk: 0 = low, 1 = high).
    """
    if filepath is None:
        # Resolve path relative to the unified project root.
        project_root = os.path.dirname(
            os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        )
        filepath = os.path.join(project_root, "data", "heart.csv")

    df = pd.read_csv(filepath)

    # Separate features and target
    X = df.drop("target", axis=1)
    y = df["target"]

    print(f"[DataLoader] Dataset loaded: {X.shape[0]} samples, {X.shape[1]} features")
    return X, y
