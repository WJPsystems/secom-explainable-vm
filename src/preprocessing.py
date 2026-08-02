"""
Preprocessing helpers for the SECOM dataset:
- missingness screening
- near-zero-variance screening
- stratified train/test splitting that preserves the fail-class ratio

These are intentionally simple, transparent functions rather than a pipeline
object, since the capstone write-up needs to report screening thresholds and
counts explicitly (see docs/data_dictionary_template.csv).
"""

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

RAW_DIR = Path(__file__).resolve().parents[1] / "data" / "raw"
MODEL_READY_DIR = Path(__file__).resolve().parents[1] / "data" / "model_ready"
MODEL_READY_DIR.mkdir(parents=True, exist_ok=True)


def load_raw() -> tuple[pd.DataFrame, pd.Series]:
    """
    Returns (features, labels) with labels remapped to {0, 1} (1 = fail,
    0 = pass), NOT SECOM's raw {-1, 1} encoding.

    This remapping used to happen only inside 02_modeling_rq1.ipynb's own
    cell, not here -- 03/04 called load_raw() directly and got raw {-1, 1}
    labels while using models trained (in 02) on 0/1 labels. AUC-ROC is
    unaffected by this (sklearn treats the larger label value as positive
    either way), but any direct equality comparison -- accuracy, McNemar's
    test -- silently breaks: {0,1}-encoded predictions almost never equal
    {-1,1}-encoded true labels even when the underlying prediction is
    correct. This produced a real, confirmed bug: 04's full-vs-reduced
    accuracy comparison came back exactly 0.0000/0.0000 on real data.
    Centralizing the remap here, rather than requiring every notebook to
    remember to do it locally, is the actual fix -- that inconsistency is
    what caused the bug in the first place.
    """
    features_path = RAW_DIR / "secom_features.csv"
    labels_path = RAW_DIR / "secom_labels.csv"

    if not features_path.exists() or not labels_path.exists():
        print("Raw SECOM data not found locally -- fetching from the UCI Machine "
              "Learning Repository now (one-time download, needs internet access)...")
        from fetch_data import fetch_secom
        fetch_secom()

    features = pd.read_csv(features_path)
    labels = pd.read_csv(labels_path).squeeze("columns")
    labels = (labels == 1).astype(int)
    return features, labels


def screen_missingness(df: pd.DataFrame, max_missing_frac: float = 0.4) -> pd.DataFrame:
    """Drop columns with more than `max_missing_frac` missing values."""
    missing_frac = df.isna().mean()
    keep_cols = missing_frac[missing_frac <= max_missing_frac].index
    dropped = df.shape[1] - len(keep_cols)
    print(f"Dropped {dropped} columns exceeding {max_missing_frac:.0%} missingness.")
    return df[keep_cols]


def screen_variance(df: pd.DataFrame, min_variance: float = 1e-6) -> pd.DataFrame:
    """Drop near-constant (near-zero-variance) columns."""
    variances = df.var(numeric_only=True)
    keep_cols = variances[variances > min_variance].index
    dropped = df.shape[1] - len(keep_cols)
    print(f"Dropped {dropped} near-zero-variance columns (threshold={min_variance}).")
    return df[keep_cols]


def impute_median(df: pd.DataFrame) -> pd.DataFrame:
    return df.fillna(df.median(numeric_only=True))


def stratified_split(
    X: pd.DataFrame, y: pd.Series, test_size: float = 0.2, random_state: int = 42
):
    """Stratified split that preserves the ~6.6% fail-class ratio in both sets."""
    return train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )


def run_screening_pipeline() -> None:
    X, y = load_raw()
    X = screen_missingness(X)
    X = screen_variance(X)
    X = impute_median(X)

    X_train, X_test, y_train, y_test = stratified_split(X, y)

    X_train.to_csv(MODEL_READY_DIR / "X_train.csv", index=False)
    X_test.to_csv(MODEL_READY_DIR / "X_test.csv", index=False)
    y_train.to_csv(MODEL_READY_DIR / "y_train.csv", index=False)
    y_test.to_csv(MODEL_READY_DIR / "y_test.csv", index=False)

    print(f"Final feature count after screening: {X.shape[1]}")
    print(f"Train: {X_train.shape[0]} rows | Test: {X_test.shape[0]} rows")


if __name__ == "__main__":
    run_screening_pipeline()
