"""
split.py
--------
Handles train/validation splitting with stratification.
Split is performed BEFORE any preprocessing to prevent data leakage.
"""

from sklearn.model_selection import train_test_split

from config import CONFIG


def split_data(X, y, test_size=None, random_state=None):
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
        Proportion of data for validation. Defaults to CONFIG['TEST_SIZE'].
    random_state : int, optional
        Random seed. Defaults to CONFIG['RANDOM_SEED'].

    Returns
    -------
    X_train : pd.DataFrame
    X_val : pd.DataFrame
    y_train : pd.Series
    y_val : pd.Series
    """
    if test_size is None:
        test_size = CONFIG["TEST_SIZE"]
    if random_state is None:
        random_state = CONFIG["RANDOM_SEED"]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"[Split] Training set: {X_train.shape[0]} samples")
    print(f"[Split] Validation set: {X_val.shape[0]} samples")

    return X_train, X_val, y_train, y_val
