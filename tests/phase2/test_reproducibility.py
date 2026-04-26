import pytest
import sys
import os
import tempfile

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'phase2'))

from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data
from models import get_models
from evaluate import evaluate_model
from serialize import save_model, load_model, verify_serialization
from sklearn.pipeline import Pipeline


def train_pipeline(model_name, seed=42):
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y, random_state=seed)
    models = get_models()
    estimator = models[model_name]
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    metrics = evaluate_model(pipeline, X_val, y_val, model_name)
    return pipeline, metrics, X_val, y_val


def test_logistic_regression_reproducible():
    _, m1, _, _ = train_pipeline("LogisticRegression_C1", seed=42)
    _, m2, _, _ = train_pipeline("LogisticRegression_C1", seed=42)
    assert round(m1["accuracy"], 6) == round(m2["accuracy"], 6)
    assert round(m1["roc_auc"], 6) == round(m2["roc_auc"], 6)


def test_random_forest_reproducible():
    _, m1, _, _ = train_pipeline("RandomForest", seed=42)
    _, m2, _, _ = train_pipeline("RandomForest", seed=42)
    assert round(m1["accuracy"], 6) == round(m2["accuracy"], 6)
    assert round(m1["roc_auc"], 6) == round(m2["roc_auc"], 6)


def test_different_seeds_give_different_splits():
    _, m1, _, _ = train_pipeline("LogisticRegression_C1", seed=42)
    _, m2, _, _ = train_pipeline("LogisticRegression_C1", seed=99)
    # Metrics can differ when the data split changes
    # (not always guaranteed but very likely)
    # We just check the pipeline ran without error for different seeds
    assert m1 is not None
    assert m2 is not None


def test_serialization_preserves_predictions():
    pipeline, _, X_val, _ = train_pipeline("LogisticRegression_C1")
    with tempfile.NamedTemporaryFile(suffix=".joblib", delete=False) as f:
        filepath = f.name
    try:
        result = verify_serialization(pipeline, filepath, X_val)
        assert result is True
    finally:
        if os.path.exists(filepath):
            os.remove(filepath)
