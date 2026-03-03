"""
test_models.py
--------------
Unit tests for the models module.
"""

import sys
import os
import pytest

# Allow imports from src/
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from models import get_models


@pytest.fixture(scope="module")
def models():
    return get_models()


def test_get_models_returns_at_least_two(models):
    """get_models() must return at least 2 model configurations."""
    assert len(models) >= 2, f"Expected ≥2 models, got {len(models)}"


def test_each_model_has_fit_method(models):
    """Each model must expose a fit() method."""
    for name, estimator in models.items():
        assert hasattr(estimator, "fit"), f"{name} has no fit() method"
        assert callable(getattr(estimator, "fit")), f"{name}.fit is not callable"


def test_each_model_has_predict_proba_method(models):
    """Each model must expose predict_proba() for confidence estimation."""
    for name, estimator in models.items():
        assert hasattr(estimator, "predict_proba"), (
            f"{name} has no predict_proba() method"
        )
        assert callable(getattr(estimator, "predict_proba")), (
            f"{name}.predict_proba is not callable"
        )
