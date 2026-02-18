"""
split.py
--------
Handles train/validation splitting with stratification.
Split is performed BEFORE any preprocessing to prevent data leakage.
"""

from sklearn.model_selection import train_test_split


def split_data(X, y, test_size=0.2, random_state=42):
    """
    Split features and target into training and validation sets.

    Uses stratified splitting to maintain class balance in both sets.
    This split must be done BEFORE any preprocessing (scaling, encoding)
    to prevent data leakage.
"""
    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Split] Training set: {X_train.shape[0]} samples")
    print(f"[Split] Validation set: {X_val.shape[0]} samples")

    return X_train, X_val, y_train, y_val
