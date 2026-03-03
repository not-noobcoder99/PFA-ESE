"""
test_evaluate.py
----------------
Unit tests for the evaluate module.
"""

import sys
import os
import pytest
import numpy as np

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from sklearn.pipeline import Pipeline
from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data
from models import get_models
from evaluate import evaluate_model


@pytest.fixture(scope="module")
def trained_pipeline():
    """Train the first available model pipeline and return it with val data."""
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    models = get_models()
    name, estimator = next(iter(models.items()))
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    return pipeline, X_val, y_val


def test_evaluate_model_returns_dict(trained_pipeline):
    """evaluate_model() must return a dict."""
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val)
    assert isinstance(metrics, dict), "evaluate_model() should return a dict"


def test_required_keys_present(trained_pipeline):
    """evaluate_model() result must contain all required metric keys."""
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val)
    required_keys = ["accuracy", "roc_auc", "f1_macro", "mcc"]
    for key in required_keys:
        assert key in metrics, f"Missing key in metrics: '{key}'"


def test_metric_values_in_range(trained_pipeline):
    """Bounded metrics must be in [0, 1] (MCC is in [-1, 1])."""
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val)
    bounded = ["accuracy", "roc_auc", "f1_macro", "precision_macro", "recall_macro"]
    for key in bounded:
        val = metrics[key]
        assert 0.0 <= val <= 1.0, f"{key}={val} is outside [0, 1]"
    # MCC is bounded by [-1, 1]
    assert -1.0 <= metrics["mcc"] <= 1.0, f"mcc={metrics['mcc']} outside [-1, 1]"


def test_confusion_matrix_shape(trained_pipeline):
    """Confusion matrix must have shape (2, 2) for binary classification."""
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val)
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2, "Confusion matrix must have 2 rows"
    assert len(cm[0]) == 2, "Confusion matrix must have 2 columns"
