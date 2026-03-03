"""
train_compare.py
----------------
Main training and comparison script for Assignment 2.

This script:
1.  Loads the centralised config
2.  Loads and splits data (using seed from config)
3.  Builds a preprocessing + model pipeline for EACH model in get_models()
4.  Trains all models
5.  Evaluates all models (multi-metric) using evaluate.py
6.  Runs error analysis for each model using error_analysis.py
7.  Compares models side-by-side and prints a comparison table
8.  Serialises each trained model to outputs/<model_name>.joblib
9.  Verifies serialisation (reloads and asserts predictions match)
10. Saves evaluation metrics to outputs/evaluation_results.json
11. Saves comparison summary to outputs/model_comparison.txt
"""

import sys
import os
import json

# Ensure src/ is on the import path when the script is run directly
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from sklearn.pipeline import Pipeline

from config import CONFIG
from data_loader import load_data
from preprocessing import build_preprocessor
from split import split_data
from models import get_models
from evaluate import evaluate_model
from error_analysis import analyse_errors
from serialize import save_model, verify_serialization


def build_pipeline(X_train, estimator):
    """
    Build an sklearn Pipeline combining a StandardScaler preprocessor
    with the given estimator.

    Parameters
    ----------
    X_train : pd.DataFrame
        Training features (used only to determine column names).
    estimator : sklearn estimator
        Unfitted classifier that will follow the preprocessor.

    Returns
    -------
    pipeline : sklearn.pipeline.Pipeline
    """
    preprocessor = build_preprocessor(X_train)
    return Pipeline([
        ("preprocessing", preprocessor),
        ("model", estimator),
    ])


def print_comparison_table(all_metrics):
    """
    Print a side-by-side comparison table for all evaluated models.

    Parameters
    ----------
    all_metrics : dict
        Keys are model names; values are metric dicts returned by
        ``evaluate_model``.
    """
    metric_keys = ["accuracy", "precision_macro", "recall_macro",
                   "f1_macro", "roc_auc", "mcc"]
    col_w = 22

    header = f"{'Metric':<20}" + "".join(
        f"{name:>{col_w}}" for name in all_metrics
    )
    print("=" * (20 + col_w * len(all_metrics)))
    print("  Model Comparison Table")
    print("=" * (20 + col_w * len(all_metrics)))
    print(header)
    print("─" * (20 + col_w * len(all_metrics)))

    for key in metric_keys:
        row = f"{key:<20}"
        for metrics in all_metrics.values():
            val = metrics.get(key, float("nan"))
            row += f"{val:>{col_w}.4f}"
        print(row)

    print("=" * (20 + col_w * len(all_metrics)))


def main():
    print("=" * 60)
    print("  Heart Disease Risk Screening – Assignment 2")
    print("  Multi-Model Training, Evaluation & Comparison")
    print("=" * 60)

    # ── [Step 1] Load config ──
    print("\n[Step 1] Loading centralised config...")
    seed = CONFIG["RANDOM_SEED"]
    test_size = CONFIG["TEST_SIZE"]
    confidence_threshold = CONFIG["CONFIDENCE_THRESHOLD"]
    print(f"  RANDOM_SEED          = {seed}")
    print(f"  TEST_SIZE            = {test_size}")
    print(f"  CONFIDENCE_THRESHOLD = {confidence_threshold}")

    # ── [Step 2] Load data & split ──
    print("\n[Step 2] Loading and splitting data...")
    X, y = load_data()
    X_train, X_val, y_train, y_val = split_data(
        X, y, test_size=test_size, random_state=seed
    )
    feature_names = X.columns.tolist()

    # ── [Step 3] Build pipelines ──
    print("\n[Step 3] Building preprocessing + model pipelines...")
    models = get_models()
    pipelines = {}
    for name, estimator in models.items():
        pipelines[name] = build_pipeline(X_train, estimator)
        print(f"  Pipeline built for: {name}")

    # ── [Step 4] Train all models ──
    print("\n[Step 4] Training all models...")
    for name, pipeline in pipelines.items():
        print(f"\n  Training: {name}")
        pipeline.fit(X_train, y_train)
        print(f"  [Done] {name} trained.")

    # ── [Step 5] Evaluate all models ──
    print("\n[Step 5] Evaluating all models (multi-metric)...")
    all_metrics = {}
    for name, pipeline in pipelines.items():
        print(f"\n  Evaluating: {name}")
        metrics = evaluate_model(pipeline, X_val, y_val, model_name=name)
        all_metrics[name] = metrics

    # ── [Step 6] Error analysis ──
    print("\n[Step 6] Running error analysis for each model...")
    all_error_analysis = {}
    for name, pipeline in pipelines.items():
        print(f"\n  Error analysis: {name}")
        analysis = analyse_errors(pipeline, X_val, y_val, feature_names)
        all_error_analysis[name] = analysis

    # ── [Step 7] Comparison table ──
    print("\n[Step 7] Model comparison table:")
    print_comparison_table(all_metrics)

    # ── [Step 8 & 9] Serialise & verify ──
    print("\n[Step 8] Serializing trained models...")
    output_dir = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "outputs"
    )
    os.makedirs(output_dir, exist_ok=True)

    for name, pipeline in pipelines.items():
        filepath = os.path.join(output_dir, f"{name}.joblib")
        print(f"\n[Step 9] Verifying serialization for: {name}")
        verify_serialization(pipeline, filepath, X_val.head(10))

    # ── [Step 10] Save evaluation_results.json ──
    print("\n[Step 10] Saving evaluation results to outputs/evaluation_results.json...")
    results_path = os.path.join(output_dir, "evaluation_results.json")
    with open(results_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"  Saved: {results_path}")

    # ── [Step 11] Save model_comparison.txt ──
    print("\n[Step 11] Saving comparison summary to outputs/model_comparison.txt...")
    comparison_path = os.path.join(output_dir, "model_comparison.txt")
    metric_keys = ["accuracy", "precision_macro", "recall_macro",
                   "f1_macro", "roc_auc", "mcc"]
    col_w = 22
    with open(comparison_path, "w") as f:
        f.write("Heart Disease Risk Screening – Model Comparison (Assignment 2)\n")
        f.write("=" * (20 + col_w * len(all_metrics)) + "\n")
        header_line = f"{'Metric':<20}" + "".join(
            f"{n:>{col_w}}" for n in all_metrics
        )
        f.write(header_line + "\n")
        f.write("─" * (20 + col_w * len(all_metrics)) + "\n")
        for key in metric_keys:
            row = f"{key:<20}"
            for metrics in all_metrics.values():
                val = metrics.get(key, float("nan"))
                row += f"{val:>{col_w}.4f}"
            f.write(row + "\n")
        f.write("=" * (20 + col_w * len(all_metrics)) + "\n")
    print(f"  Saved: {comparison_path}")

    print("\n" + "=" * 60)
    print("  Assignment 2 pipeline execution complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
