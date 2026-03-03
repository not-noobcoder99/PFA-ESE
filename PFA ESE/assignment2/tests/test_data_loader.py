"""
test_data_loader.py
-------------------
Unit tests for the data_loader module.
"""

import sys
import os
import pytest
import pandas as pd

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_data


@pytest.fixture(scope="module")
def dataset():
    """Load the dataset once and share across tests."""
    X, y = load_data()
    return X, y


def test_load_data_returns_dataframe_and_series(dataset):
    """load_data() must return a DataFrame and a Series."""
    X, y = dataset
    assert isinstance(X, pd.DataFrame), "X should be a pd.DataFrame"
    assert isinstance(y, pd.Series), "y should be a pd.Series"


def test_feature_matrix_has_13_columns(dataset):
    """Feature matrix must have exactly 13 columns."""
    X, _ = dataset
    assert X.shape[1] == 13, f"Expected 13 features, got {X.shape[1]}"


def test_target_contains_only_binary_values(dataset):
    """Target variable must only contain values 0 and 1."""
    _, y = dataset
    unique_vals = set(y.unique())
    assert unique_vals <= {0, 1}, f"Unexpected target values: {unique_vals}"


def test_no_missing_values(dataset):
    """Dataset must have no missing (NaN) values."""
    X, y = dataset
    assert X.isnull().sum().sum() == 0, "X contains missing values"
    assert y.isnull().sum() == 0, "y contains missing values"
