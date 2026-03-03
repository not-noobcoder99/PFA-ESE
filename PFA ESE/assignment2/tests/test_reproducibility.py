"""
test_reproducibility.py
-----------------------
Tests that results are identical across runs with the same seed,
differ with different seeds, and that serialised model predictions
match the in-memory model predictions.
"""

import sys
import os
import tempfile
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
from serialize import save_model, load_model


def _train_and_evaluate(model_name, seed):
    """Helper: train a named model with the given seed and return metrics."""
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y, random_state=seed)
    models = get_models()
    estimator = models[model_name]
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_val, y_val)
    return pipeline, metrics, X_val


def test_logistic_regression_same_seed_identical():
    """Logistic Regression trained twice with the same seed must give identical metrics."""
    _, metrics1, _ = _train_and_evaluate("LogisticRegression", seed=42)
    _, metrics2, _ = _train_and_evaluate("LogisticRegression", seed=42)
    for key in ["accuracy", "roc_auc", "f1_macro", "mcc"]:
        assert metrics1[key] == metrics2[key], (
            f"LogisticRegression metric '{key}' differs across runs with same seed: "
            f"{metrics1[key]} vs {metrics2[key]}"
        )


def test_random_forest_same_seed_identical():
    """Random Forest trained twice with the same seed must give identical metrics."""
    _, metrics1, _ = _train_and_evaluate("RandomForest", seed=42)
    _, metrics2, _ = _train_and_evaluate("RandomForest", seed=42)
    for key in ["accuracy", "roc_auc", "f1_macro", "mcc"]:
        assert metrics1[key] == metrics2[key], (
            f"RandomForest metric '{key}' differs across runs with same seed: "
            f"{metrics1[key]} vs {metrics2[key]}"
        )


def test_different_seed_may_give_different_metrics():
    """
    Sanity check: using a very different seed should (in general) change
    the validation split and thus at least one metric.

    This may theoretically pass even with the same metrics if the dataset
    split happens to yield the same score, but in practice it differs.
    """
    _, metrics_42, _ = _train_and_evaluate("LogisticRegression", seed=42)
    _, metrics_99, _ = _train_and_evaluate("LogisticRegression", seed=99)
    # At least ONE metric should differ when the split changes
    changed = any(
        metrics_42[k] != metrics_99[k]
        for k in ["accuracy", "roc_auc", "f1_macro", "mcc"]
    )
    assert changed, (
        "Expected at least one metric to change with a different random seed "
        "(sanity check failed – all metrics were identical with seed=42 and seed=99)"
    )


def test_serialised_model_predictions_match():
    """Saved and reloaded model must produce identical predictions to in-memory model."""
    pipeline, _, X_val = _train_and_evaluate("LogisticRegression", seed=42)

    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        save_model(pipeline, tmp_path)
        loaded = load_model(tmp_path)

        preds_orig = pipeline.predict(X_val)
        preds_load = loaded.predict(X_val)
        assert np.array_equal(preds_orig, preds_load), (
            "predict() outputs differ between original and reloaded model"
        )

        proba_orig = pipeline.predict_proba(X_val)
        proba_load = loaded.predict_proba(X_val)
        assert np.allclose(proba_orig, proba_load), (
            "predict_proba() outputs differ between original and reloaded model"
        )
    finally:
        os.unlink(tmp_path)
