import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data
from models import get_models
from evaluate import evaluate_model
from sklearn.pipeline import Pipeline


@pytest.fixture(scope="module")
def trained_pipeline():
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(X, y)
    models = get_models()
    name = list(models.keys())[0]
    estimator = models[name]
    preprocessor = build_preprocessor(X_train)
    pipeline = Pipeline([("preprocessing", preprocessor), ("model", estimator)])
    pipeline.fit(X_train, y_train)
    return pipeline, X_val, y_val


def test_evaluate_returns_dict(trained_pipeline):
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val, "test_model")
    assert isinstance(metrics, dict)


def test_required_keys_present(trained_pipeline):
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val, "test_model")
    for key in ["accuracy", "roc_auc", "f1_macro", "mcc", "precision_macro", "recall_macro"]:
        assert key in metrics, f"Missing key: {key}"


def test_metrics_in_valid_range(trained_pipeline):
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val, "test_model")
    for key in ["accuracy", "roc_auc", "f1_macro", "precision_macro", "recall_macro"]:
        assert 0.0 <= metrics[key] <= 1.0, f"{key} out of range: {metrics[key]}"


def test_confusion_matrix_shape(trained_pipeline):
    pipeline, X_val, y_val = trained_pipeline
    metrics = evaluate_model(pipeline, X_val, y_val, "test_model")
    cm = metrics["confusion_matrix"]
    assert len(cm) == 2 and len(cm[0]) == 2
