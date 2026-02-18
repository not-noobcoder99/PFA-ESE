# Assignment 1 – Heart Disease Risk Screening

## Course: DS201 – Programming for AI

## Project: Confidence-Aware ML Pipeline for Heart Disease Risk Screening

---

### Overview

This project implements a **confidence-aware machine learning pipeline** for heart disease risk screening. The system not only predicts heart disease risk but also computes a **confidence score** for each prediction and **defers uncertain predictions** to ensure patient safety.

### Dataset

- **Source**: UCI Heart Disease (Cleveland) Dataset
- **Samples**: 303 patient records
- **Features**: 13 clinical attributes (age, cholesterol, blood pressure, etc.)
- **Target**: Binary (0 = Low Risk, 1 = High Risk)

### Project Structure

```
assignment1/
│
├── data/
│   └── heart.csv              # Heart Disease dataset
│
├── src/
│   ├── data_loader.py          # Loads and separates features/target
│   ├── preprocessing.py        # StandardScaler pipeline (prevents leakage)
│   ├── split.py                # Stratified train/validation split
│   ├── model.py                # Logistic Regression baseline model
│   └── train.py                # Main training + confidence-aware logic
│
├── outputs/
│   └── baseline_metrics.txt    # Saved evaluation metrics
│
└── README.md
```

### How to Run

```bash
cd assignment1/src
python train.py
```

### Requirements

- Python 3.8+
- pandas
- numpy
- scikit-learn
- seaborn (for EDA)
- matplotlib (for EDA)

Install dependencies:
```bash
pip install pandas numpy scikit-learn seaborn matplotlib
```

### Key Features

1. **Modular Design**: Each component (loading, preprocessing, splitting, modeling) is in a separate file
2. **Data Leakage Prevention**: Split is done BEFORE preprocessing; scaling fitted only on training data
3. **Confidence-Aware Predictions**: Uses `predict_proba()` to compute confidence scores
4. **Decision Deferral**: Predictions with confidence < 0.70 are deferred for human review
5. **Pipeline Integration**: All steps embedded in an sklearn Pipeline for reproducibility

### Confidence Logic

```
confidence = max(predicted_probabilities)

If confidence >= 0.70 → Prediction Accepted
If confidence <  0.70 → Prediction Deferred
```

### Ethical Considerations

- This system is a **screening tool**, not a medical diagnosis system
- Uncertain predictions are flagged for **human review**
- The system prioritizes **patient safety** over accuracy
- No personal identifiers are used in the dataset
