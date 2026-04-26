import pytest
import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'phase2'))

from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data


def test_preprocessor_builds_without_error():
    X, y = load_data()
    preprocessor = build_preprocessor(X)
    assert preprocessor is not None


def test_preprocessor_output_shape():
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    preprocessor = build_preprocessor(X_train)
    X_transformed = preprocessor.fit_transform(X_train)
    assert X_transformed.shape == X_train.shape


def test_scaled_values_near_zero_mean():
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    preprocessor = build_preprocessor(X_train)
    X_scaled = preprocessor.fit_transform(X_train)
    means = np.abs(X_scaled.mean(axis=0))
    assert np.all(means < 1e-9), "Scaled training data should have near-zero mean"
