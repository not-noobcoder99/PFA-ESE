"""
train_compare.py
----------------
Main script for Assignment 2.
Trains and compares multiple sklearn pipelines on the Heart Disease dataset.
Evaluates each model, performs error analysis, serialises models to disk,
and saves results to the outputs/ directory.
"""

import sys
import os
import json

# Ensure src/ modules are importable when run from any working directory
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import data_loader
import split as split_module
import models as models_module
import evaluate as evaluate_module
import error_analysis as error_analysis_module
import serialize as serialize_module
from preprocessing import build_preprocessor
from config import RANDOM_SEED, OUTPUT_DIR
from sklearn.pipeline import Pipeline


def main():
    """
    End-to-end training, evaluation, and comparison pipeline for Assignment 2.

    Steps
    -----
    1. Setup output directory.
    2. Load data.
    3. Stratified train/validation split.
    4. Build, train, evaluate, and analyse each model.
    5. Print side-by-side comparison table.
    6. Serialise each model and verify serialisation.
    7. Save results to outputs/.
    """
    # ── Step 1: Setup ─────────────────────────────────────────────────────
    print("=" * 60)
    print("  ASSIGNMENT 2 — MULTI-MODEL COMPARISON")
    print("=" * 60)
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # ── Step 2: Load data ──────────────────────────────────────────────────
    print("\n[Step 1] Loading data...")
    X, y = data_loader.load_data()

    # ── Step 3: Split ──────────────────────────────────────────────────────
    print("\n[Step 2] Splitting data...")
    X_train, X_val, y_train, y_val = split_module.split_data(X, y, random_state=RANDOM_SEED)

    # ── Step 4 & 5: Build, train, and evaluate each model ─────────────────
    all_models = models_module.get_models()
    evaluation_results = {}
    trained_pipelines = {}
    feature_names = X_train.columns.tolist()

    for step_num, (model_name, estimator) in enumerate(all_models.items(), start=3):
        print(f"\n[Step {step_num}] Training {model_name}...")
        preprocessor = build_preprocessor(X_train)
        pipeline = Pipeline([
            ("preprocessing", preprocessor),
            ("model", estimator),
        ])
        pipeline.fit(X_train, y_train)
        trained_pipelines[model_name] = pipeline

        # Evaluate
        metrics = evaluate_module.evaluate_model(pipeline, X_val, y_val, model_name)
        evaluation_results[model_name] = metrics

        # Error analysis
        error_analysis_module.analyse_errors(pipeline, X_val, y_val, feature_names)

    # ── Step 6: Comparison table ───────────────────────────────────────────
    print("\n" + "=" * 60)
    print("  MODEL COMPARISON TABLE")
    print("=" * 60)
    header_cols = ["Model", "Acc", "Prec", "Rec", "F1", "AUC", "MCC"]
    col_w = [24, 7, 7, 7, 7, 7, 7]
    header = "  ".join(f"{h:<{w}}" for h, w in zip(header_cols, col_w))
    print(header)
    print("─" * 72)
    for model_name, metrics in evaluation_results.items():
        row = [
            model_name,
            f"{metrics['accuracy']:.4f}",
            f"{metrics['precision_macro']:.4f}",
            f"{metrics['recall_macro']:.4f}",
            f"{metrics['f1_macro']:.4f}",
            f"{metrics['roc_auc']:.4f}",
            f"{metrics['mcc']:.4f}",
        ]
        print("  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)))
    print("=" * 60)

    # ── Step 7: Serialise and verify each model ────────────────────────────
    step_offset = 3 + len(all_models)
    for i, (model_name, pipeline) in enumerate(trained_pipelines.items(), start=step_offset):
        print(f"\n[Step {i}] Serialising {model_name}...")
        filepath = os.path.join(OUTPUT_DIR, f"{model_name}.joblib")
        serialize_module.verify_serialization(pipeline, filepath, X_val)

    # ── Step 8: Save results ───────────────────────────────────────────────
    results_path = os.path.join(OUTPUT_DIR, "evaluation_results.json")
    with open(results_path, "w", encoding="utf-8") as f:
        json.dump(evaluation_results, f, indent=2)
    print(f"\n[Results] Evaluation metrics saved to: {results_path}")

    comparison_path = os.path.join(OUTPUT_DIR, "model_comparison.txt")
    with open(comparison_path, "w", encoding="utf-8") as f:
        f.write("MODEL COMPARISON TABLE\n")
        f.write("=" * 72 + "\n")
        f.write("  ".join(f"{h:<{w}}" for h, w in zip(header_cols, col_w)) + "\n")
        f.write("─" * 72 + "\n")
        for model_name, metrics in evaluation_results.items():
            row = [
                model_name,
                f"{metrics['accuracy']:.4f}",
                f"{metrics['precision_macro']:.4f}",
                f"{metrics['recall_macro']:.4f}",
                f"{metrics['f1_macro']:.4f}",
                f"{metrics['roc_auc']:.4f}",
                f"{metrics['mcc']:.4f}",
            ]
            f.write("  ".join(f"{v:<{w}}" for v, w in zip(row, col_w)) + "\n")
        f.write("=" * 72 + "\n")
    print(f"[Results] Comparison table saved to: {comparison_path}")

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()
