# Assignment 2 — Multi-Model Comparison & Evaluation

**Course:** PFA ESE  
**Assignment:** 2 — Comparative Model Analysis on Heart Disease Data

---

## Overview

Assignment 2 extends Assignment 1 by introducing a structured multi-model comparison framework. Three sklearn pipelines are trained on the UCI Heart Disease dataset:

1. **Logistic Regression (C=1.0)** — default regularisation strength
2. **Logistic Regression (C=0.1)** — stronger L2 regularisation (tuned variant)
3. **Random Forest (100 trees, max_depth=5)** — ensemble non-linear model

Each pipeline is evaluated with a comprehensive suite of metrics, subjected to error analysis, serialised to disk, and verified for reproducibility.

---

## Directory Structure

```
assignment2/
├── data/
│   └── heart.csv              # UCI Heart Disease dataset (303 records)
├── src/
│   ├── config.py              # Central config: seeds, thresholds, hyperparams, paths
│   ├── data_loader.py         # Load heart.csv → (X, y)
│   ├── preprocessing.py       # StandardScaler inside ColumnTransformer
│   ├── split.py               # Stratified train/validation split
│   ├── models.py              # Dict of named unfitted estimators
│   ├── evaluate.py            # Comprehensive metric computation + report printing
│   ├── error_analysis.py      # FP/FN analysis with feature-level breakdown
│   ├── serialize.py           # joblib save/load + verification
│   └── train_compare.py       # Main script: orchestrates all steps
├── tests/
│   ├── __init__.py
│   ├── test_data_loader.py
│   ├── test_preprocessing.py
│   ├── test_models.py
│   ├── test_evaluate.py
│   └── test_reproducibility.py
├── outputs/                   # Created at runtime
│   ├── .gitkeep
│   ├── evaluation_results.json
│   ├── model_comparison.txt
│   ├── LogisticRegression_C1.joblib
│   ├── LogisticRegression_C01.joblib
│   └── RandomForest.joblib
├── requirements.txt
└── README.md
```

---

## Models Used and Trade-offs

### Logistic Regression
- **Pros:** Interpretable coefficients, fast training, calibrated probabilities, strong medical-context baseline.
- **Cons:** Assumes linear decision boundary; can underfit complex feature interactions.
- **C=1.0 vs C=0.1:** Lower C applies stronger L2 regularisation, reducing variance at the cost of some bias — useful when overfitting is suspected on small datasets.

### Random Forest
- **Pros:** Captures non-linear interactions, robust to outliers, provides feature importances, generally stronger on tabular data.
- **Cons:** Less interpretable than logistic regression, more hyperparameters, slower to train, probability calibration can be less reliable.

---

## Evaluation Metrics

| Metric | Why it matters for medical screening |
|---|---|
| **Accuracy** | Overall correctness; useful baseline measure |
| **Precision (macro)** | Average positive predictive value — controls false alarm rates |
| **Recall (macro)** | Average sensitivity — critical: missing a high-risk patient is costly |
| **F1 (macro)** | Harmonic mean of precision & recall; balanced measure on imbalanced classes |
| **ROC-AUC** | Rank-ordering ability across thresholds; threshold-independent |
| **MCC** | Matthews Correlation Coefficient — robust single-number summary for binary classification, handles class imbalance better than accuracy |
| **Confusion Matrix** | Raw breakdown of TP/TN/FP/FN for clinical interpretation |

In medical screening, **recall for the positive class (High Risk)** is especially important — failing to identify a high-risk patient (False Negative) has higher clinical cost than a false alarm (False Positive).

---

## Reproducibility Controls

All seeds, split ratios, and hyperparameters are defined in a single `src/config.py`:

```python
RANDOM_SEED = 42
TEST_SIZE = 0.2
CONFIDENCE_THRESHOLD = 0.70
```

- The train/validation split uses `stratify=y` with `random_state=RANDOM_SEED`.
- All estimators receive `random_state=RANDOM_SEED` where applicable.
- No magic numbers appear outside `config.py`.

---

## How to Run

### Train and compare all models
```bash
cd assignment2/src
python train_compare.py
```

### Run tests
```bash
cd assignment2
pytest tests/ -v
```

---

## Expected Outputs

After running `train_compare.py`, the `outputs/` directory will contain:

| File | Description |
|---|---|
| `evaluation_results.json` | All metrics for all models in JSON format |
| `model_comparison.txt` | Side-by-side comparison table |
| `LogisticRegression_C1.joblib` | Serialised LR (C=1) pipeline |
| `LogisticRegression_C01.joblib` | Serialised LR (C=0.1) pipeline |
| `RandomForest.joblib` | Serialised Random Forest pipeline |

---

## Engineering Trade-offs

- **Pipeline design:** Embedding `StandardScaler` inside `sklearn.Pipeline` ensures the scaler is always fitted on training data only, preventing data leakage during cross-validation or future retraining.
- **Centralised config:** `config.py` is the single source of truth for all hyperparameters and paths. This makes hyperparameter experiments traceable and reproducible.
- **joblib serialisation:** joblib is preferred over pickle for numpy-heavy sklearn objects due to efficient memory mapping of large arrays.
- **ColumnTransformer:** Used even though all features are numeric, to allow future extension (e.g., adding categorical encoders) without restructuring the pipeline.

---

## Reflection on Evaluation Findings

With only 303 samples, all models are expected to show relatively similar performance. Logistic Regression with C=1.0 provides a strong baseline due to the near-linear nature of the decision boundary in this dataset. The tuned variant (C=0.1) may show slightly lower variance on this small dataset. Random Forest, while more expressive, may show marginal improvement in ROC-AUC given the limited data. The error analysis highlights which features most consistently differ between misclassified and correctly classified samples, providing clinical insight into hard cases.
