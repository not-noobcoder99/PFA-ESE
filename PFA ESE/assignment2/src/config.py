"""
config.py
---------
Centralised reproducibility configuration for Assignment 2.

All random seeds, hyperparameters, and thresholds are defined here.
Every other module imports from this file — no magic numbers scattered around.
"""

CONFIG = {
    # ── Reproducibility ──
    "RANDOM_SEED": 42,

    # ── Data split ──
    "TEST_SIZE": 0.2,

    # ── Confidence threshold ──
    "CONFIDENCE_THRESHOLD": 0.70,

    # ── Model hyperparameters ──
    "LOGISTIC_REGRESSION": {
        "C": 1.0,
        "max_iter": 1000,
        "random_state": 42,
    },
    "LOGISTIC_REGRESSION_L2": {
        "C": 0.1,
        "max_iter": 1000,
        "random_state": 42,
    },
    "RANDOM_FOREST": {
        "n_estimators": 100,
        "max_depth": 5,
        "random_state": 42,
    },
}
