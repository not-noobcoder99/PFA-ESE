import pytest
import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..', 'src', 'phase2'))

from models import get_models


def test_get_models_returns_dict():
    models = get_models()
    assert isinstance(models, dict)


def test_at_least_two_models():
    models = get_models()
    assert len(models) >= 2


def test_all_models_have_fit():
    models = get_models()
    for name, model in models.items():
        assert hasattr(model, 'fit'), f"{name} missing fit()"


def test_all_models_have_predict_proba():
    models = get_models()
    for name, model in models.items():
        assert hasattr(model, 'predict_proba'), f"{name} missing predict_proba()"


def test_model_names_are_strings():
    models = get_models()
    for name in models.keys():
        assert isinstance(name, str)
