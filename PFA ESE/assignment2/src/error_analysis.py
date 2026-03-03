"""
error_analysis.py
-----------------
Systematic error analysis for a trained classification pipeline.

Identifies False Positives and False Negatives, computes mean feature
values for each error group, and highlights which features differ most
from correctly classified samples.
"""

import numpy as np
import pandas as pd


def analyse_errors(pipeline, X_val, y_val, feature_names):
    """
    Perform systematic error analysis on validation predictions.

    Identifies False Positives (predicted High Risk, actually Low Risk)
    and False Negatives (predicted Low Risk, actually High Risk), then
    computes mean feature values for each group and compares them to
    correctly classified samples to highlight the largest deviations.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline containing a preprocessor and a classifier.
    X_val : pd.DataFrame or np.ndarray
        Validation feature matrix.
    y_val : pd.Series or np.ndarray
        True labels for the validation set.
    feature_names : list of str
        Names of the feature columns (used for reporting).

    Returns
    -------
    analysis : dict
        Structured dict with keys:
        ``false_positives``, ``false_negatives``, ``correct``,
        ``fp_mean_features``, ``fn_mean_features``,
        ``correct_mean_features``, ``top_fp_features``,
        ``top_fn_features``.
    """
    y_pred = pipeline.predict(X_val)
    y_true = np.array(y_val)

    # Convert X_val to DataFrame for easy indexing
    if not isinstance(X_val, pd.DataFrame):
        X_df = pd.DataFrame(X_val, columns=feature_names)
    else:
        X_df = X_val.reset_index(drop=True)

    y_true_arr = np.array(y_true).ravel()
    y_pred_arr = np.array(y_pred).ravel()

    # ── Index masks ──
    fp_mask = (y_pred_arr == 1) & (y_true_arr == 0)   # False Positives
    fn_mask = (y_pred_arr == 0) & (y_true_arr == 1)   # False Negatives
    correct_mask = y_pred_arr == y_true_arr             # Correctly classified

    fp_df = X_df[fp_mask]
    fn_df = X_df[fn_mask]
    correct_df = X_df[correct_mask]

    # ── Mean feature values per group ──
    fp_mean = fp_df.mean() if len(fp_df) > 0 else pd.Series(
        np.zeros(len(feature_names)), index=feature_names
    )
    fn_mean = fn_df.mean() if len(fn_df) > 0 else pd.Series(
        np.zeros(len(feature_names)), index=feature_names
    )
    correct_mean = correct_df.mean() if len(correct_df) > 0 else pd.Series(
        np.zeros(len(feature_names)), index=feature_names
    )

    # ── Top diverging features (absolute diff from correct mean) ──
    fp_diff = (fp_mean - correct_mean).abs().sort_values(ascending=False)
    fn_diff = (fn_mean - correct_mean).abs().sort_values(ascending=False)

    top_fp = fp_diff.head(5).index.tolist()
    top_fn = fn_diff.head(5).index.tolist()

    # ── Printed report ──
    print("=" * 60)
    print("  Error Analysis Report")
    print("=" * 60)
    print(f"  Total validation samples : {len(y_true_arr)}")
    print(f"  Correctly classified     : {correct_mask.sum()}")
    print(f"  False Positives (FP)     : {fp_mask.sum()}  "
          "(predicted High Risk, actually Low Risk)")
    print(f"  False Negatives (FN)     : {fn_mask.sum()}  "
          "(predicted Low Risk, actually High Risk)  ← critical")

    print("\n─" + "─" * 39)
    print("  Top features diverging in False Positives:")
    for feat in top_fp:
        print(f"    {feat:<20}  FP mean={fp_mean[feat]:.3f}  "
              f"Correct mean={correct_mean[feat]:.3f}  "
              f"Δ={fp_diff[feat]:.3f}")

    print("\n─" + "─" * 39)
    print("  Top features diverging in False Negatives:")
    for feat in top_fn:
        print(f"    {feat:<20}  FN mean={fn_mean[feat]:.3f}  "
              f"Correct mean={correct_mean[feat]:.3f}  "
              f"Δ={fn_diff[feat]:.3f}")
    print("=" * 60)

    analysis = {
        "false_positives": int(fp_mask.sum()),
        "false_negatives": int(fn_mask.sum()),
        "correct": int(correct_mask.sum()),
        "fp_mean_features": fp_mean.to_dict(),
        "fn_mean_features": fn_mean.to_dict(),
        "correct_mean_features": correct_mean.to_dict(),
        "top_fp_features": top_fp,
        "top_fn_features": top_fn,
    }

    return analysis
