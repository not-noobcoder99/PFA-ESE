"""
evaluate.py
-----------
Comprehensive multi-metric model evaluation for Assignment 2.

Computes accuracy, precision, recall, F1, ROC-AUC, confusion matrix,
and Matthews Correlation Coefficient (MCC) for a trained pipeline.
"""

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    matthews_corrcoef,
    classification_report,
)


def evaluate_model(pipeline, X_val, y_val, model_name="Model"):
    """
    Compute and print a comprehensive evaluation report for a trained pipeline.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline containing a preprocessor and a classifier.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series or np.ndarray
        True labels for the validation set.
    model_name : str, optional
        Human-readable label used in the printed report.

    Returns
    -------
    metrics : dict
        Dictionary with keys:
        ``accuracy``, ``precision_macro``, ``precision_per_class``,
        ``recall_macro``, ``recall_per_class``, ``f1_macro``,
        ``f1_per_class``, ``roc_auc``, ``confusion_matrix``, ``mcc``.
    """
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    # ── Core metrics ──
    accuracy = accuracy_score(y_val, y_pred)
    precision_macro = precision_score(y_val, y_pred, average="macro", zero_division=0)
    precision_per_class = precision_score(
        y_val, y_pred, average=None, zero_division=0
    ).tolist()
    recall_macro = recall_score(y_val, y_pred, average="macro", zero_division=0)
    recall_per_class = recall_score(
        y_val, y_pred, average=None, zero_division=0
    ).tolist()
    f1_macro = f1_score(y_val, y_pred, average="macro", zero_division=0)
    f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0).tolist()
    roc_auc = roc_auc_score(y_val, y_proba)
    cm = confusion_matrix(y_val, y_pred)
    mcc = matthews_corrcoef(y_val, y_pred)
    report = classification_report(
        y_val, y_pred, target_names=["Low Risk", "High Risk"]
    )

    # ── Formatted report ──
    print("=" * 60)
    print(f"  Evaluation Report – {model_name}")
    print("=" * 60)
    print(f"  Accuracy       : {accuracy:.4f}")
    print(f"  Precision(mac) : {precision_macro:.4f}")
    print(f"  Recall(macro)  : {recall_macro:.4f}")
    print(f"  F1-Score(mac)  : {f1_macro:.4f}")
    print(f"  ROC-AUC        : {roc_auc:.4f}")
    print(f"  MCC            : {mcc:.4f}")
    print("─" * 40)
    print(f"  Per-Class Precision : Low={precision_per_class[0]:.4f}  High={precision_per_class[1]:.4f}")
    print(f"  Per-Class Recall    : Low={recall_per_class[0]:.4f}  High={recall_per_class[1]:.4f}")
    print(f"  Per-Class F1        : Low={f1_per_class[0]:.4f}  High={f1_per_class[1]:.4f}")
    print("─" * 40)
    print(f"  Confusion Matrix:\n{cm}")
    print("─" * 40)
    print(f"  Classification Report:\n{report}")

    metrics = {
        "accuracy": accuracy,
        "precision_macro": precision_macro,
        "precision_per_class": precision_per_class,
        "recall_macro": recall_macro,
        "recall_per_class": recall_per_class,
        "f1_macro": f1_macro,
        "f1_per_class": f1_per_class,
        "roc_auc": roc_auc,
        "confusion_matrix": cm.tolist(),
        "mcc": mcc,
    }

    return metrics
