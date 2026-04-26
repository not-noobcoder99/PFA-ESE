
"""
train.py
--------
Main training script for the Heart Disease Risk Screening pipeline.

This script:
1. Loads the dataset
2. Splits data BEFORE preprocessing (prevents data leakage)
3. Builds a preprocessing + model pipeline
4. Trains the baseline Logistic Regression model
5. Evaluates on validation set
6. Applies confidence-aware prediction logic
7. Saves metrics to outputs/
"""

import sys
import os
import json
import numpy as np
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# Add src/ directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from data_loader import load_data
from preprocessing import build_preprocessor
from model import get_model
from split import split_data


# ──────────────────────────────────────────────
# Configuration
# ──────────────────────────────────────────────
CONFIDENCE_THRESHOLD = 0.70  # Predictions below this are deferred


def confidence_aware_predict(pipeline, X, threshold=CONFIDENCE_THRESHOLD):
    """
    Make predictions with confidence-aware decision logic.

    For each sample:
      - Compute class probabilities
      - confidence = max(predicted_probabilities)
      - If confidence >= threshold → Prediction Accepted
      - If confidence <  threshold → Prediction Deferred

    Parameters
    ----------
    pipeline : sklearn.pipeline.Pipeline
        Trained pipeline (preprocessor + model).
    X : pd.DataFrame
        Feature matrix for prediction.
    threshold : float
        Confidence threshold for accepting predictions.

    Returns
    -------
    results : list of dict
        Each dict contains: risk, confidence, decision, and optional note.
    """
    probabilities = pipeline.predict_proba(X)
    results = []

    for i, probs in enumerate(probabilities):
        confidence = float(np.max(probs))
        predicted_class = int(np.argmax(probs))

        if confidence >= threshold:
            results.append({
                "sample_index": i,
                "risk": "High" if predicted_class == 1 else "Low",
                "confidence": round(confidence, 4),
                "decision": "Accepted"
            })
        else:
            results.append({
                "sample_index": i,
                "risk": None,
                "confidence": round(confidence, 4),
                "decision": "Deferred",
                "note": "Further medical evaluation recommended"
            })

    return results


def main():
    print("=" * 60)
    print("  Heart Disease Risk Screening – Assignment 1")
    print("  Confidence-Aware ML Pipeline")
    print("=" * 60)

    # ── Step 1: Load Data ──
    print("\n[Step 1] Loading dataset...")
    X, y = load_data()

    # ── Step 2: Train/Validation Split (BEFORE preprocessing) ──
    print("\n[Step 2] Splitting data (before preprocessing to prevent leakage)...")
    X_train, X_val, y_train, y_val = split_data(X, y)

    # ── Step 3: Build Pipeline (Preprocessor + Model) ──
    print("\n[Step 3] Building preprocessing + model pipeline...")
    preprocessor = build_preprocessor(X_train)
    model = get_model()

    pipeline = Pipeline([
        ("preprocessing", preprocessor),
        ("model", model)
    ])
    print("[Pipeline] Preprocessor: StandardScaler | Model: LogisticRegression")

    # ── Step 4: Train ──
    print("\n[Step 4] Training model on training set...")
    pipeline.fit(X_train, y_train)
    print("[Training] Model training complete.")

    # ── Step 5: Standard Evaluation ──
    print("\n[Step 5] Evaluating on validation set...")
    preds = pipeline.predict(X_val)
    acc = accuracy_score(y_val, preds)
    report = classification_report(y_val, preds, target_names=["Low Risk", "High Risk"])
    cm = confusion_matrix(y_val, preds)

    print(f"\n  Validation Accuracy: {acc:.4f}")
    print(f"\n  Classification Report:\n{report}")
    print(f"  Confusion Matrix:\n{cm}\n")

    # ── Step 6: Confidence-Aware Predictions ──
    print("\n[Step 6] Applying confidence-aware prediction logic...")
    print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
    results = confidence_aware_predict(pipeline, X_val)

    accepted = [r for r in results if r["decision"] == "Accepted"]
    deferred = [r for r in results if r["decision"] == "Deferred"]

    print(f"\n  Total predictions:    {len(results)}")
    print(f"  Accepted predictions: {len(accepted)}")
    print(f"  Deferred predictions: {len(deferred)}")
    print(f"  Deferral rate:        {len(deferred) / len(results) * 100:.1f}%")

    # ── Show Sample Outputs ──
    print("\n── Sample Accepted Prediction ──")
    if accepted:
        print(json.dumps(accepted[0], indent=2))

    print("\n── Sample Deferred Prediction ──")
    if deferred:
        print(json.dumps(deferred[0], indent=2))
    else:
        print("  No deferred predictions (all above threshold)")

    # ── Step 7: Calculate Accuracy on Accepted Predictions Only ──
    accepted_indices = [r["sample_index"] for r in accepted]
    if accepted_indices:
        y_val_arr = y_val.values
        accepted_preds = preds[accepted_indices]
        accepted_true = y_val_arr[accepted_indices]
        accepted_acc = accuracy_score(accepted_true, accepted_preds)
        print(f"\n  Accuracy on ACCEPTED predictions only: {accepted_acc:.4f}")
    else:
        accepted_acc = None
        print("\n  No accepted predictions to evaluate.")

    # ── Step 8: Save Metrics ──
    print("\n[Step 7] Saving metrics to outputs/phase1/baseline_metrics.txt...")
    project_root = os.path.dirname(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    )
    output_dir = os.path.join(project_root, "outputs", "phase1")
    os.makedirs(output_dir, exist_ok=True)

    metrics_path = os.path.join(output_dir, "baseline_metrics.txt")
    with open(metrics_path, "w") as f:
        f.write("Heart Disease Risk Screening – Baseline Metrics (Assignment 1)\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Model: Logistic Regression\n")
        f.write(f"Confidence Threshold: {CONFIDENCE_THRESHOLD}\n\n")
        f.write(f"Validation Accuracy: {acc:.4f}\n\n")
        f.write(f"Classification Report:\n{report}\n")
        f.write(f"Confusion Matrix:\n{cm}\n\n")
        f.write(f"Total Predictions:    {len(results)}\n")
        f.write(f"Accepted Predictions: {len(accepted)}\n")
        f.write(f"Deferred Predictions: {len(deferred)}\n")
        f.write(f"Deferral Rate:        {len(deferred) / len(results) * 100:.1f}%\n")
        if accepted_acc is not None:
            f.write(f"Accuracy on Accepted: {accepted_acc:.4f}\n")

    print(f"[Done] Metrics saved to: {metrics_path}")
    print("\n" + "=" * 60)
    print("  Pipeline execution complete.")
    print("=" * 60)


if __name__ == "__main__":
    main()
