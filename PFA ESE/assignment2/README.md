# Assignment 2 – Evaluation, Testing & Reproducibility

## Project Overview

This assignment builds on the Assignment 1 Heart Disease Risk Screening
pipeline by introducing:

- **Multiple model configurations** (Logistic Regression, Random Forest)
- **Comprehensive multi-metric evaluation** (Accuracy, Precision, Recall,
  F1, ROC-AUC, Confusion Matrix, MCC)
- **Systematic error analysis** (False Positive / False Negative patterns
  with feature-level breakdown)
- **Reproducibility controls** (centralised config, fixed seeds)
- **Model serialization** (joblib save/load + round-trip verification)
- **Full pytest test suite**

---

## Assignment 2 Objectives

1. Centralise all random seeds and hyperparameters in `src/config.py`
2. Train and compare at least two distinct model types in a single run
3. Compute 8+ evaluation metrics for every model
4. Identify FP/FN error patterns with per-feature breakdown
5. Persist trained models with joblib and verify round-trip integrity (serialization)
6. Ensure all tests pass with `pytest tests/ -v`
7. Eliminate data leakage (preprocessing always fit on training data only)

---

## Directory Structure

```
assignment2/
│
├── data/
│   └── heart.csv                  # UCI Heart Disease (Cleveland) dataset
│
├── src/
│   ├── config.py                  # Centralised config (seeds, hyperparams)
│   ├── data_loader.py             # Dataset loading
│   ├── preprocessing.py           # StandardScaler pipeline
│   ├── split.py                   # Stratified train/val split
│   ├── models.py                  # Two+ model configurations
│   ├── evaluate.py                # Multi-metric evaluation
│   ├── error_analysis.py          # FP/FN error analysis
│   ├── train_compare.py           # Main training & comparison script
│   └── serialize.py               # joblib serialization utilities
│
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluate.py
│   └── test_reproducibility.py
│
├── outputs/                       # Generated artifacts (gitkeep)
├── requirements.txt
└── README.md
```

---

## How to Run

### Install dependencies

```bash
pip install -r requirements.txt
```

### Train and compare models

```bash
cd src
python train_compare.py
```

### Run all tests

```bash
pytest tests/ -v
```

---

## Models

| Model | Hyperparameters | Rationale |
|---|---|---|
| **Logistic Regression** | `C=1.0`, `max_iter=1000`, `random_state=42` | Interpretable baseline; outputs calibrated probabilities; widely used in clinical ML |
| **Random Forest** | `n_estimators=100`, `max_depth=5`, `random_state=42` | Ensemble method; captures non-linear feature interactions; less prone to outliers |
| **LogReg (C=0.1)** | `C=0.1`, `max_iter=1000`, `random_state=42` | Stronger L2 regularisation; useful for high-variance data or many features |

---

## Evaluation Metrics

| Metric | Why it matters for medical screening |
|---|---|
| **Accuracy** | Overall correctness; useful only when classes are balanced |
| **Precision** | Fraction of predicted positives that are true positives; reduces unnecessary follow-up |
| **Recall / Sensitivity** | Fraction of true positives caught; **critical** – missing a high-risk patient is dangerous |
| **F1-Score** | Harmonic mean of precision & recall; balances both concerns |
| **ROC-AUC** | Discrimination ability across all thresholds; threshold-independent |
| **Confusion Matrix** | Full breakdown of TP/TN/FP/FN; essential for understanding error types |
| **MCC** | Matthews Correlation Coefficient; robust single-value metric for imbalanced data |

---

## Reproducibility Controls

- All seeds are defined in `src/config.py` (`RANDOM_SEED = 42`)
- `split_data()` accepts `random_state` from config
- All model constructors use `random_state` from config
- `test_reproducibility.py` asserts identical metrics across repeated runs with the same seed

---

## Example Output

```
============================================================
  Model Comparison Table
============================================================
Metric                LogisticRegression       RandomForest  LogisticRegression_C0.1
────────────────────────────────────────────────────────────────────────────────────
accuracy                        0.8689           0.8689                   0.8689
precision_macro                 0.8696           0.8731                   0.8645
recall_macro                    0.8676           0.8601                   0.8601
f1_macro                        0.8684           0.8655                   0.8617
roc_auc                         0.9315           0.9282                   0.9206
mcc                             0.7361           0.7372                   0.7261
============================================================
```
