"""
serialize.py
------------
Handles saving and loading of trained sklearn pipelines using joblib.
Provides serialisation verification to ensure saved models produce
identical predictions after reloading.
"""

import numpy as np
import joblib


def save_model(pipeline, filepath):
    """
    Save a trained sklearn pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline to save.
    filepath : str
        Destination file path (e.g. 'outputs/model.joblib').
    """
    joblib.dump(pipeline, filepath)
    print(f"[Serialize] Model saved to: {filepath}")


def load_model(filepath):
    """
    Load a trained sklearn pipeline from disk using joblib.

    Parameters
    ----------
    filepath : str
        Path to the saved joblib file.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The loaded pipeline.
    """
    pipeline = joblib.load(filepath)
    print(f"[Serialize] Model loaded from: {filepath}")
    return pipeline


def verify_serialization(original_pipeline, filepath, X_sample):
    """
    Verify that a saved and re-loaded pipeline produces identical predictions.

    Saves the original pipeline to filepath, reloads it, and compares
    predict_proba outputs using numpy's assert_array_almost_equal.

    Parameters
    ----------
    original_pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline to verify.
    filepath : str
        File path used for saving and loading the pipeline.
    X_sample : pd.DataFrame or np.ndarray
        Sample data to run predictions on.

    Returns
    -------
    verified : bool
        True if predictions match after serialisation.

    Raises
    ------
    AssertionError
        If predictions differ between original and reloaded pipeline.
    """
    save_model(original_pipeline, filepath)
    reloaded_pipeline = load_model(filepath)

    original_proba = original_pipeline.predict_proba(X_sample)
    reloaded_proba = reloaded_pipeline.predict_proba(X_sample)

    try:
        np.testing.assert_array_almost_equal(original_proba, reloaded_proba)
        print(f"[Serialize] Verification PASSED: predictions are identical after serialisation.")
        return True
    except AssertionError as e:
        print(f"[Serialize] Verification FAILED: {e}")
        raise
