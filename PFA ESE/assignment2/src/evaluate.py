"""
evaluate.py
-----------
Evaluates a trained sklearn pipeline on validation data.
Computes a comprehensive set of metrics appropriate for medical screening tasks.
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


def evaluate_model(pipeline, X_val, y_val, model_name):
    """
    Evaluate a trained sklearn pipeline on validation data.

    Computes accuracy, precision, recall, F1 (macro and per-class),
    ROC-AUC, confusion matrix, and Matthews Correlation Coefficient.
    Prints a formatted report and returns all metrics as a dictionary.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted sklearn pipeline with preprocessing and model steps.
    X_val : pd.DataFrame
        Validation feature matrix.
    y_val : pd.Series
        True validation labels.
    model_name : str
        Name of the model for display in the report.

    Returns
    -------
    metrics : dict
        Dictionary containing all computed metrics:
        - accuracy, precision_macro, recall_macro, f1_macro
        - roc_auc, mcc
        - precision_class0, precision_class1
        - recall_class0, recall_class1
        - f1_class0, f1_class1
        - confusion_matrix (nested list for JSON serialisation)
    """
    y_pred = pipeline.predict(X_val)
    y_proba = pipeline.predict_proba(X_val)[:, 1]

    acc = accuracy_score(y_val, y_pred)
    prec_macro = precision_score(y_val, y_pred, average='macro', zero_division=0)
    rec_macro = recall_score(y_val, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_val, y_pred, average='macro', zero_division=0)
    roc_auc = roc_auc_score(y_val, y_proba)
    mcc = matthews_corrcoef(y_val, y_pred)

    prec_per_class = precision_score(y_val, y_pred, average=None, zero_division=0)
    rec_per_class = recall_score(y_val, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_val, y_pred, average=None, zero_division=0)
    cm = confusion_matrix(y_val, y_pred)

    # ── Print formatted report ──────────────────────────────────────────────
    print("=" * 60)
    print(f"  EVALUATION REPORT: {model_name}")
    print("=" * 60)
    print(f"  Accuracy          : {acc:.4f}")
    print(f"  ROC-AUC           : {roc_auc:.4f}")
    print(f"  MCC               : {mcc:.4f}")
    print("─" * 40)
    print(f"  Precision (macro) : {prec_macro:.4f}")
    print(f"  Recall    (macro) : {rec_macro:.4f}")
    print(f"  F1        (macro) : {f1_macro:.4f}")
    print("─" * 40)
    print(f"  Precision Class 0 : {prec_per_class[0]:.4f}  |  Class 1 : {prec_per_class[1]:.4f}")
    print(f"  Recall    Class 0 : {rec_per_class[0]:.4f}  |  Class 1 : {rec_per_class[1]:.4f}")
    print(f"  F1        Class 0 : {f1_per_class[0]:.4f}  |  Class 1 : {f1_per_class[1]:.4f}")
    print("─" * 40)
    print("  Confusion Matrix:")
    print(f"    TN={cm[0,0]:>4}  FP={cm[0,1]:>4}")
    print(f"    FN={cm[1,0]:>4}  TP={cm[1,1]:>4}")
    print("=" * 60)

    metrics = {
        "accuracy": float(acc),
        "precision_macro": float(prec_macro),
        "recall_macro": float(rec_macro),
        "f1_macro": float(f1_macro),
        "roc_auc": float(roc_auc),
        "mcc": float(mcc),
        "precision_class0": float(prec_per_class[0]),
        "precision_class1": float(prec_per_class[1]),
        "recall_class0": float(rec_per_class[0]),
        "recall_class1": float(rec_per_class[1]),
        "f1_class0": float(f1_per_class[0]),
        "f1_class1": float(f1_per_class[1]),
        "confusion_matrix": cm.tolist(),
    }

    return metrics
