import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'phase2'))

import pandas as pd
from data_loader import load_data


def test_load_data_returns_correct_types():
    X, y = load_data()
    assert isinstance(X, pd.DataFrame)
    assert isinstance(y, pd.Series)


def test_feature_matrix_has_13_columns():
    X, y = load_data()
    assert X.shape[1] == 13


def test_target_binary():
    X, y = load_data()
    assert set(y.unique()).issubset({0, 1})


def test_no_missing_values():
    X, y = load_data()
    assert X.isnull().sum().sum() == 0
    assert y.isnull().sum() == 0


def test_dataset_size():
    X, y = load_data()
    assert X.shape[0] == 303
    assert len(y) == 303
