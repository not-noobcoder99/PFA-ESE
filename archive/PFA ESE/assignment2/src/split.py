"""
split.py
--------
Handles train/validation splitting with stratification.
Split is performed BEFORE any preprocessing to prevent data leakage.
"""

from sklearn.model_selection import train_test_split
from config import TEST_SIZE, RANDOM_SEED


def split_data(X, y, test_size=TEST_SIZE, random_state=RANDOM_SEED):
    """
    Split features and target into training and validation sets.

    Uses stratified splitting to maintain class balance in both sets.
    This split must be done BEFORE any preprocessing (scaling, encoding)
    to prevent data leakage.

    Parameters
    ----------
    X : pd.DataFrame
        Feature matrix.
    y : pd.Series
        Target variable.
    test_size : float, optional
        Proportion of the dataset to include in the validation split.
        Defaults to TEST_SIZE from config.py.
    random_state : int, optional
        Random seed for reproducibility. Defaults to RANDOM_SEED from config.py.

    Returns
    -------
    X_train : pd.DataFrame
        Training feature matrix.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_train : pd.Series
        Training target variable.
    y_val : pd.Series
        Validation target variable.
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
