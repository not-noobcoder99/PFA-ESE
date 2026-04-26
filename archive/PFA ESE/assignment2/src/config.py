"""
config.py
---------
Centralised reproducibility configuration for Assignment 2.
All hyperparameters, seeds, and thresholds are defined here.
No magic numbers in any other file.
"""

# ── Reproducibility ──
RANDOM_SEED = 42
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.70

# ── Model Hyperparameters ──
LOGISTIC_REGRESSION_CONFIG = {
    "C": 1.0,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

LOGISTIC_REGRESSION_TUNED_CONFIG = {
    "C": 0.1,
    "max_iter": 1000,
    "random_state": RANDOM_SEED,
    "solver": "lbfgs",
}

RANDOM_FOREST_CONFIG = {
    "n_estimators": 100,
    "max_depth": 5,
    "random_state": RANDOM_SEED,
    "n_jobs": -1,
}

# ── Paths ──
import os
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
