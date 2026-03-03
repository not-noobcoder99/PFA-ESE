"""
serialize.py
------------
Model serialisation and deserialisation using joblib.

Provides save, load, and round-trip verification utilities so that
trained pipelines can be persisted to disk and later reloaded with
guaranteed identical predictions.
"""

import numpy as np
import joblib


def save_model(pipeline, filepath):
    """
    Persist a trained sklearn pipeline to disk using joblib.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline to serialise.
    filepath : str
        Destination file path (e.g. 'outputs/model.joblib').

    Returns
    -------
    None
    """
    joblib.dump(pipeline, filepath)
    print(f"[Serialize] Model saved to: {filepath}")


def load_model(filepath):
    """
    Load a previously saved sklearn pipeline from disk.

    Parameters
    ----------
    filepath : str
        Path to the .joblib file produced by ``save_model``.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
        The deserialised pipeline, ready for inference.
    """
    pipeline = joblib.load(filepath)
    print(f"[Serialize] Model loaded from: {filepath}")
    return pipeline


def verify_serialization(original_pipeline, filepath, X_sample):
    """
    Verify that a serialised model produces identical predictions to the
    in-memory model.

    Saves ``original_pipeline`` to ``filepath``, loads it back, then
    asserts that both ``predict`` and ``predict_proba`` outputs are
    numerically identical.

    Parameters
    ----------
    original_pipeline : sklearn.pipeline.Pipeline
        The in-memory trained pipeline.
    filepath : str
        Path where the model will be saved (and loaded from).
    X_sample : pd.DataFrame or np.ndarray
        A small sample of feature rows used to compare predictions.

    Returns
    -------
    None

    Raises
    ------
    AssertionError
        If predictions from the reloaded model differ from the original.
    """
    save_model(original_pipeline, filepath)
    loaded_pipeline = load_model(filepath)

    preds_original = original_pipeline.predict(X_sample)
    preds_loaded = loaded_pipeline.predict(X_sample)

    proba_original = original_pipeline.predict_proba(X_sample)
    proba_loaded = loaded_pipeline.predict_proba(X_sample)

    assert np.array_equal(preds_original, preds_loaded), (
        "predict() outputs differ between original and reloaded model!"
    )
    assert np.allclose(proba_original, proba_loaded), (
        "predict_proba() outputs differ between original and reloaded model!"
    )

    print("[Serialize] Verification passed: original and reloaded models produce identical predictions.")
