"""
error_analysis.py
-----------------
Analyses model errors by identifying false positives and false negatives,
and comparing their mean feature values against correctly classified samples.
"""

import numpy as np
import pandas as pd


def analyse_errors(pipeline, X_val, y_val, feature_names):
    """
    Perform error analysis on a trained pipeline's validation predictions.

    Identifies false positives (predicted high-risk but actually low-risk) and
    false negatives (predicted low-risk but actually high-risk). Computes mean
    feature values for each error group and for correct predictions, then ranks
    features by their discriminating power between error groups and correct group.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted sklearn pipeline.
    X_val : pd.DataFrame or np.ndarray
        Validation feature matrix.
    y_val : pd.Series or np.ndarray
        True validation labels.
    feature_names : list of str
        Names of features corresponding to columns of X_val.

    Returns
    -------
    result : dict
        Dictionary with keys:
        - n_false_positives (int)
        - n_false_negatives (int)
        - n_correct (int)
        - fp_feature_means (dict): mean feature values for FP samples
        - fn_feature_means (dict): mean feature values for FN samples
        - correct_feature_means (dict): mean feature values for correct samples
        - fp_top_differentiating_features (list of (feature, diff) tuples)
        - fn_top_differentiating_features (list of (feature, diff) tuples)
    """
    y_pred = pipeline.predict(X_val)
    y_true = np.array(y_val)

    # Convert X_val to numpy for indexing
    if isinstance(X_val, pd.DataFrame):
        X_arr = X_val.values
    else:
        X_arr = np.array(X_val)

    fp_mask = (y_pred == 1) & (y_true == 0)
    fn_mask = (y_pred == 0) & (y_true == 1)
    correct_mask = (y_pred == y_true)

    fp_indices = np.where(fp_mask)[0]
    fn_indices = np.where(fn_mask)[0]
    correct_indices = np.where(correct_mask)[0]

    n_fp = int(fp_mask.sum())
    n_fn = int(fn_mask.sum())
    n_correct = int(correct_mask.sum())
    n_total = len(y_true)

    # Compute mean feature values for each group
    def _mean_dict(indices):
        if len(indices) == 0:
            return {f: 0.0 for f in feature_names}
        means = X_arr[indices].mean(axis=0)
        return {f: float(v) for f, v in zip(feature_names, means)}

    fp_means = _mean_dict(fp_indices)
    fn_means = _mean_dict(fn_indices)
    correct_means = _mean_dict(correct_indices)

    # Rank features by absolute difference between FP/FN and correct means
    fp_diffs = sorted(
        [(f, abs(fp_means[f] - correct_means[f])) for f in feature_names],
        key=lambda x: x[1],
        reverse=True,
    )
    fn_diffs = sorted(
        [(f, abs(fn_means[f] - correct_means[f])) for f in feature_names],
        key=lambda x: x[1],
        reverse=True,
    )

    # ── Print formatted report ──────────────────────────────────────────────
    print("=" * 60)
    print("  ERROR ANALYSIS")
    print("=" * 60)
    print(f"  Total validation samples : {n_total}")
    print(f"  Correct predictions      : {n_correct} ({100*n_correct/n_total:.1f}%)")
    print(f"  False Positives          : {n_fp} ({100*n_fp/n_total:.1f}%)  [predicted High, actual Low]")
    print(f"  False Negatives          : {n_fn} ({100*n_fn/n_total:.1f}%)  [predicted Low, actual High]")
    print("─" * 40)
    print("  Top-5 features differentiating FP from correct:")
    for feat, diff in fp_diffs[:5]:
        print(f"    {feat:<15} Δ = {diff:.4f}")
    print("─" * 40)
    print("  Top-5 features differentiating FN from correct:")
    for feat, diff in fn_diffs[:5]:
        print(f"    {feat:<15} Δ = {diff:.4f}")
    print("=" * 60)

    return {
        "n_false_positives": n_fp,
        "n_false_negatives": n_fn,
        "n_correct": n_correct,
        "fp_feature_means": fp_means,
        "fn_feature_means": fn_means,
        "correct_feature_means": correct_means,
        "fp_top_differentiating_features": fp_diffs,
        "fn_top_differentiating_features": fn_diffs,
    }
