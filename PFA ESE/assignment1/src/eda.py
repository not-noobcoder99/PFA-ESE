
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import os
import sys

# ──────────────────────────────────────────────
# Setup
# ──────────────────────────────────────────────
# Resolve paths relative to the project root
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_PATH = os.path.join(BASE_DIR, "data", "heart.csv")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Set plot style
sns.set_theme(style="whitegrid")
plt.rcParams["figure.dpi"] = 120


def main():
    print("=" * 60)
    print("  Heart Disease Dataset – Exploratory Data Analysis (EDA)")
    print("=" * 60)

    # ── 1. Load Dataset ──
    df = pd.read_csv(DATA_PATH)

    # ── 2. Dataset Shape and Data Types ──
    print("\n" + "─" * 40)
    print("1. DATASET SHAPE AND DATA TYPES")
    print("─" * 40)
    print(f"\n  Shape: {df.shape[0]} rows × {df.shape[1]} columns")
    print(f"\n  Columns and Data Types:")
    for col in df.columns:
        print(f"    {col:12s} → {df[col].dtype}")

    # ── 3. Summary Statistics ──
    print("\n" + "─" * 40)
    print("2. SUMMARY STATISTICS")
    print("─" * 40)
    stats = df.describe().T
    stats["range"] = stats["max"] - stats["min"]
    print(f"\n{stats.to_string()}")

    # ── 4. Missing Values Check ──
    print("\n" + "─" * 40)
    print("3. MISSING VALUES CHECK")
    print("─" * 40)
    missing = df.isnull().sum()
    total_missing = missing.sum()
    if total_missing == 0:
        print("\n  ✓ No missing values found in any column.")
    else:
        print(f"\n  ⚠ Total missing values: {total_missing}")
        print(missing[missing > 0])

    # ── 5. Target Class Distribution ──
    print("\n" + "─" * 40)
    print("4. TARGET CLASS DISTRIBUTION")
    print("─" * 40)
    target_counts = df["target"].value_counts()
    print(f"\n  Low Risk  (0): {target_counts.get(0, 0)} samples")
    print(f"  High Risk (1): {target_counts.get(1, 0)} samples")
    print(f"  Balance Ratio:  {target_counts.min() / target_counts.max():.2f}")

    # Plot: Target Class Distribution
    fig, ax = plt.subplots(figsize=(6, 4))
    colors = ["#4CAF50", "#F44336"]
    bars = ax.bar(["Low Risk (0)", "High Risk (1)"],
                  [target_counts.get(0, 0), target_counts.get(1, 0)],
                  color=colors, edgecolor="black", linewidth=0.8)
    for bar, count in zip(bars, [target_counts.get(0, 0), target_counts.get(1, 0)]):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 2,
                str(count), ha="center", fontweight="bold", fontsize=12)
    ax.set_title("Target Class Distribution", fontsize=14, fontweight="bold")
    ax.set_ylabel("Number of Patients", fontsize=11)
    ax.set_xlabel("Heart Disease Risk", fontsize=11)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_target_distribution.png"))
    plt.close()
    print("  [Saved] eda_target_distribution.png")

    # ── 6. Feature Scale Differences ──
    print("\n" + "─" * 40)
    print("5. FEATURE SCALE DIFFERENCES")
    print("─" * 40)
    feature_cols = df.columns.drop("target")
    print(f"\n  {'Feature':12s} {'Min':>8s} {'Max':>8s} {'Mean':>10s} {'Std':>10s}")
    print(f"  {'─'*50}")
    for col in feature_cols:
        print(f"  {col:12s} {df[col].min():8.2f} {df[col].max():8.2f} "
              f"{df[col].mean():10.2f} {df[col].std():10.2f}")

    # Plot: Feature Scale Comparison (Box Plot)
    fig, ax = plt.subplots(figsize=(12, 5))
    df[feature_cols].boxplot(ax=ax, patch_artist=True,
                              boxprops=dict(facecolor="#90CAF9", edgecolor="black"),
                              medianprops=dict(color="red", linewidth=2))
    ax.set_title("Feature Scale Differences (Before Scaling)", fontsize=14, fontweight="bold")
    ax.set_ylabel("Value", fontsize=11)
    ax.set_xlabel("Features", fontsize=11)
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_feature_scales.png"))
    plt.close()
    print("\n  [Saved] eda_feature_scales.png")
    print("\n  Observation: Features like 'chol' (up to 564) and 'trestbps' (up to 200)")
    print("  have much larger scales than 'fbs' (0–1) and 'sex' (0–1).")
    print("  → StandardScaler is required to ensure fair model learning.")

    # ── 7. Correlation Heatmap ──
    print("\n" + "─" * 40)
    print("6. CORRELATION HEATMAP")
    print("─" * 40)
    corr = df.corr()
    fig, ax = plt.subplots(figsize=(10, 8))
    mask = np.triu(np.ones_like(corr, dtype=bool))
    sns.heatmap(corr, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.5,
                cbar_kws={"shrink": 0.8}, ax=ax)
    ax.set_title("Feature Correlation Heatmap", fontsize=14, fontweight="bold")
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_correlation_heatmap.png"))
    plt.close()
    print("  [Saved] eda_correlation_heatmap.png")

    # Key correlations with target
    target_corr = corr["target"].drop("target").sort_values(ascending=False)
    print(f"\n  Top correlations with target:")
    for feat, val in target_corr.items():
        direction = "+" if val > 0 else "−"
        print(f"    {feat:12s} → {direction}{abs(val):.3f}")

    # ── 8. Feature Distributions by Target Class ──
    print("\n" + "─" * 40)
    print("7. FEATURE DISTRIBUTIONS BY TARGET CLASS")
    print("─" * 40)
    continuous_features = ["age", "trestbps", "chol", "thalach", "oldpeak"]

    fig, axes = plt.subplots(2, 3, figsize=(14, 8))
    axes = axes.flatten()
    for i, feat in enumerate(continuous_features):
        ax = axes[i]
        for label, color, name in [(0, "#4CAF50", "Low Risk"), (1, "#F44336", "High Risk")]:
            subset = df[df["target"] == label][feat]
            ax.hist(subset, bins=20, alpha=0.6, color=color, label=name, edgecolor="black", linewidth=0.5)
        ax.set_title(feat, fontsize=12, fontweight="bold")
        ax.set_xlabel(feat)
        ax.set_ylabel("Count")
        ax.legend(fontsize=8)
    # Remove unused subplot
    axes[-1].set_visible(False)
    fig.suptitle("Continuous Feature Distributions by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_feature_distributions.png"))
    plt.close()
    print("  [Saved] eda_feature_distributions.png")

    # ── 9. Categorical Feature Analysis ──
    print("\n" + "─" * 40)
    print("8. CATEGORICAL FEATURE ANALYSIS")
    print("─" * 40)
    categorical_features = ["sex", "cp", "fbs", "restecg", "exang", "slope", "ca", "thal"]

    fig, axes = plt.subplots(2, 4, figsize=(16, 8))
    axes = axes.flatten()
    for i, feat in enumerate(categorical_features):
        ax = axes[i]
        ct = pd.crosstab(df[feat], df["target"])
        ct.columns = ["Low Risk", "High Risk"]
        ct.plot(kind="bar", ax=ax, color=["#4CAF50", "#F44336"],
                edgecolor="black", linewidth=0.5)
        ax.set_title(feat, fontsize=11, fontweight="bold")
        ax.set_xlabel("")
        ax.set_ylabel("Count")
        ax.legend(fontsize=7)
        ax.tick_params(axis='x', rotation=0)
    fig.suptitle("Categorical Features by Target Class", fontsize=14, fontweight="bold", y=1.01)
    plt.tight_layout()
    plt.savefig(os.path.join(OUTPUT_DIR, "eda_categorical_features.png"))
    plt.close()
    print("  [Saved] eda_categorical_features.png")

    # ── 10. EDA Summary ──
    print("\n" + "=" * 60)
    print("  EDA SUMMARY")
    print("=" * 60)
    print(f"""
  • Dataset: {df.shape[0]} samples, {df.shape[1] - 1} features + 1 target
  • Missing values: {total_missing}
  • Target balance: {target_counts.get(0, 0)} Low Risk vs {target_counts.get(1, 0)} High Risk (ratio {target_counts.min() / target_counts.max():.2f})
  • Features vary significantly in magnitude → Scaling required
  • Most features are numeric (int64/float64)
  • Key positive correlations with target: cp, thalach, slope
  • Key negative correlations with target: exang, oldpeak, ca, thal

  → EDA guided preprocessing decisions:
    1. StandardScaler needed for fair learning
    2. No imputation needed (no missing data)
    3. Classes are moderately balanced (no resampling needed)
    """)

    # Save EDA summary to text file
    summary_path = os.path.join(OUTPUT_DIR, "eda_summary.txt")
    with open(summary_path, "w", encoding="utf-8") as f:
        f.write("Heart Disease Dataset – EDA Summary\n")
        f.write("=" * 60 + "\n\n")
        f.write(f"Dataset Shape: {df.shape[0]} rows × {df.shape[1]} columns\n")
        f.write(f"Features: {df.shape[1] - 1}\n")
        f.write(f"Target Variable: 'target' (0 = Low Risk, 1 = High Risk)\n\n")
        f.write(f"Missing Values: {total_missing}\n\n")
        f.write(f"Target Distribution:\n")
        f.write(f"  Low Risk  (0): {target_counts.get(0, 0)}\n")
        f.write(f"  High Risk (1): {target_counts.get(1, 0)}\n")
        f.write(f"  Balance Ratio: {target_counts.min() / target_counts.max():.2f}\n\n")
        f.write(f"Summary Statistics:\n{stats.to_string()}\n\n")
        f.write(f"Correlations with Target:\n")
        for feat, val in target_corr.items():
            f.write(f"  {feat:12s} → {val:+.3f}\n")
        f.write(f"\nKey Observations:\n")
        f.write(f"  - Features vary significantly in scale → StandardScaler required\n")
        f.write(f"  - No missing values → No imputation needed\n")
        f.write(f"  - Classes moderately balanced → No resampling needed\n")
        f.write(f"  - cp, thalach, slope positively correlated with heart disease risk\n")
        f.write(f"  - exang, oldpeak, ca, thal negatively correlated with heart disease risk\n")
    print(f"  [Saved] eda_summary.txt")
    print("\n  All EDA plots saved to outputs/ folder.")
    print("=" * 60)


if __name__ == "__main__":
    main()
