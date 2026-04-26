"""
model.py
--------
Defines the baseline machine learning model.
Logistic Regression is chosen for interpretability and probability output support.
"""

from sklearn.linear_model import LogisticRegression


def get_model():
    """
    Return a Logistic Regression classifier.

    Justification:
    - Produces class probabilities (needed for confidence estimation)
    - Interpretable coefficients
    - Computationally efficient
    - Widely used in healthcare ML research
    - Strong baseline before exploring complex models

    Returns
    -------
    model : LogisticRegression
        An unfitted Logistic Regression model.
    """
    return LogisticRegression(max_iter=1000, random_state=42)
