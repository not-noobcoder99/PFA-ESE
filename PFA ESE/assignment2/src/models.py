"""
models.py
---------
Defines all model configurations for Assignment 2.

Provides at least two distinct classifiers:
  1. Logistic Regression (C=1.0)  – interpretable baseline
  2. Random Forest (n_estimators=100, max_depth=5) – ensemble method
  3. Logistic Regression (C=0.1)  – stronger L2 regularisation

All hyperparameters are loaded from config.py.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from config import CONFIG


def get_models():
    """
    Return a dictionary of model name → unfitted estimator.

    All hyperparameters are sourced from CONFIG so that every seed
    and tuning knob is defined in a single place.

    Returns
    -------
    models : dict
        Keys are human-readable model names; values are unfitted sklearn
        estimators, each of which supports both ``fit`` and
        ``predict_proba``.

    Examples
    --------
    >>> models = get_models()
    >>> list(models.keys())
    ['LogisticRegression', 'RandomForest', 'LogisticRegression_C0.1']
    """
    lr_cfg = CONFIG["LOGISTIC_REGRESSION"]
    lr_l2_cfg = CONFIG["LOGISTIC_REGRESSION_L2"]
    rf_cfg = CONFIG["RANDOM_FOREST"]

    models = {
        "LogisticRegression": LogisticRegression(
            C=lr_cfg["C"],
            max_iter=lr_cfg["max_iter"],
            random_state=lr_cfg["random_state"],
        ),
        "RandomForest": RandomForestClassifier(
            n_estimators=rf_cfg["n_estimators"],
            max_depth=rf_cfg["max_depth"],
            random_state=rf_cfg["random_state"],
        ),
        "LogisticRegression_C0.1": LogisticRegression(
            C=lr_l2_cfg["C"],
            max_iter=lr_l2_cfg["max_iter"],
            random_state=lr_l2_cfg["random_state"],
        ),
    }

    return models
