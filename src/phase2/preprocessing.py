"""
preprocessing.py
-----------------
Builds a preprocessing pipeline using sklearn.
Handles numerical feature scaling using StandardScaler.
Designed to be used inside a full sklearn Pipeline to prevent data leakage.
"""

from sklearn.preprocessing import StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline


def build_preprocessor(X):
    """
    Build a preprocessing transformer for the heart disease dataset.

    All features are treated as numeric and scaled using StandardScaler.
    This preprocessor is meant to be embedded inside an sklearn Pipeline
    so that scaling is fitted ONLY on training data (preventing data leakage).

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix (used to extract column names).

    Returns
    -------
    preprocessor : ColumnTransformer
        An unfitted preprocessing transformer ready to be used in a Pipeline.
    """
    numeric_features = X.columns.tolist()

    numeric_pipeline = Pipeline([
        ("scaler", StandardScaler())
    ])

    preprocessor = ColumnTransformer([
        ("num", numeric_pipeline, numeric_features)
    ])

    return preprocessor
