"""
test_preprocessing.py
---------------------
Unit tests for the preprocessing module.
"""

import sys
import os
import pytest
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data


@pytest.fixture(scope="module")
def train_val_split():
    """Load and split data once for the module."""
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    return X_train, X_val, y_train, y_val


def test_build_preprocessor_no_error(train_val_split):
    """build_preprocessor() should not raise any exception."""
    X_train, _, _, _ = train_val_split
    preprocessor = build_preprocessor(X_train)
    assert preprocessor is not None


def test_output_shape_matches_input(train_val_split):
    """After fit_transform, the output row count must match input row count."""
    X_train, _, _, _ = train_val_split
    preprocessor = build_preprocessor(X_train)
    X_transformed = preprocessor.fit_transform(X_train)
    assert X_transformed.shape[0] == X_train.shape[0], (
        "Row count changed after preprocessing"
    )
    assert X_transformed.shape[1] == X_train.shape[1], (
        "Column count changed after preprocessing"
    )


def test_scaled_values_mean_near_zero(train_val_split):
    """After StandardScaler, each feature mean should be near 0."""
    X_train, _, _, _ = train_val_split
    preprocessor = build_preprocessor(X_train)
    X_transformed = preprocessor.fit_transform(X_train)
    col_means = np.mean(X_transformed, axis=0)
    assert np.allclose(col_means, 0, atol=1e-6), (
        f"Column means not near 0: {col_means}"
    )


def test_scaled_values_std_near_one(train_val_split):
    """After StandardScaler, each feature std should be near 1."""
    X_train, _, _, _ = train_val_split
    preprocessor = build_preprocessor(X_train)
    X_transformed = preprocessor.fit_transform(X_train)
    col_stds = np.std(X_transformed, axis=0)
    assert np.allclose(col_stds, 1, atol=1e-1), (
        f"Column stds not near 1: {col_stds}"
    )
