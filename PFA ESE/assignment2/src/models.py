"""
models.py
---------
Defines multiple model configurations for comparison in Assignment 2.
All hyperparameters are sourced from config.py.
"""

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from config import (
    LOGISTIC_REGRESSION_CONFIG,
    LOGISTIC_REGRESSION_TUNED_CONFIG,
    RANDOM_FOREST_CONFIG,
)


def get_models():
    """
    Return a dictionary of named model configurations.

    Returns
    -------
    models : dict
        Keys are model names (str), values are unfitted sklearn estimators.
    """
    return {
        "LogisticRegression_C1": LogisticRegression(**LOGISTIC_REGRESSION_CONFIG),
        "LogisticRegression_C01": LogisticRegression(**LOGISTIC_REGRESSION_TUNED_CONFIG),
        "RandomForest": RandomForestClassifier(**RANDOM_FOREST_CONFIG),
    }
