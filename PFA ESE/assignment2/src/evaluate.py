"""
evaluate.py
-----------
Evaluates a trained sklearn pipeline on validation data.
Computes a comprehensive set of metrics appropriate for medical screening tasks.

Classes
-------
ModelEvaluator
    Object-oriented interface for evaluating a trained sklearn pipeline.
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


class ModelEvaluator:
    """
    Evaluates a trained sklearn pipeline with a comprehensive set of metrics
    appropriate for medical screening tasks.

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        A fitted sklearn pipeline with preprocessing and model steps.
    model_name : str
        Human-readable name of the model, used in printed reports.

    Attributes
    ----------
    pipeline : sklearn.pipeline.Pipeline
        The fitted pipeline passed at construction.
    model_name : str
        Display name of the model.
    _metrics : dict or None
        Cached metrics dictionary from the most recent call to :meth:`evaluate`.
        ``None`` until :meth:`evaluate` has been called.

    Examples
    --------
    >>> evaluator = ModelEvaluator(pipeline, "LogisticRegression")
    >>> metrics = evaluator.evaluate(X_val, y_val)
    >>> print(evaluator.metrics["roc_auc"])
    """

    def __init__(self, pipeline, model_name):
        self.pipeline = pipeline
        self.model_name = model_name
        self._metrics = None

    def evaluate(self, X_val, y_val):
        """
        Evaluate the pipeline on validation data and cache the result.

        Computes accuracy, precision, recall, F1 (macro and per-class),
        ROC-AUC, confusion matrix, and Matthews Correlation Coefficient.
        Prints a formatted report and returns all metrics as a dictionary.

        Parameters
        ----------
        X_val : pd.DataFrame
            Validation feature matrix.
        y_val : pd.Series
            True validation labels.

        Returns
        -------
        metrics : dict
            Dictionary containing all computed metrics:
            ``accuracy``, ``precision_macro``, ``recall_macro``, ``f1_macro``,
            ``roc_auc``, ``mcc``, ``precision_class0``, ``precision_class1``,
            ``recall_class0``, ``recall_class1``, ``f1_class0``, ``f1_class1``,
            ``confusion_matrix`` (nested list for JSON serialisation).
        """
        self._metrics = evaluate_model(self.pipeline, X_val, y_val, self.model_name)
        return self._metrics

    @property
    def metrics(self):
        """
        Return the cached metrics from the most recent :meth:`evaluate` call.

        Raises
        ------
        RuntimeError
            If :meth:`evaluate` has not been called yet.
        """
        if self._metrics is None:
            raise RuntimeError("Call evaluate() before accessing metrics.")
        return self._metrics


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
