"""
preprocessing.py
────────────────
All data transformation steps between raw CSV and model-ready arrays.

Design decisions:
- Random Forest is a tree-based, non-parametric model.
  It does NOT benefit from feature scaling and can actually be harmed
  by it (masking relative feature magnitudes). Raw features are returned
  for tree-based models.
- SVM (LinearSVC) and MLP are sensitive to feature scale.
  StandardScaler is fit exclusively on training data to prevent
  data leakage into the test set.
- Label encoding uses sklearn's LabelEncoder so class indices are
  consistent across all downstream code.
- Stratified split preserves per-class proportions in both splits,
  which is critical with the severe class imbalance present in IDS2018.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

RANDOM_STATE = 42


# ── Column identification helpers ────────────────────────────────────────────

def _find_label_col(df: pd.DataFrame) -> str:
    """Return the name of the label column (case-insensitive match)."""
    for col in df.columns:
        if col.strip().lower() == "label":
            return col
    raise ValueError("No 'Label' column found. Check column names in your CSV.")


def _timestamp_cols(df: pd.DataFrame) -> list[str]:
    """Identify timestamp-like columns by name heuristic."""
    return [c for c in df.columns if "timestamp" in c.lower()]


# ── Main preprocessing steps ─────────────────────────────────────────────────

def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Remove non-informative and invalid rows/columns.

    Steps:
    1. Drop timestamp columns — they are identifiers, not features.
    2. Replace ±Inf with NaN, then drop rows containing NaN.
       Inf values arise from division-by-zero in flow statistics.
    """
    ts_cols = _timestamp_cols(df)
    if ts_cols:
        df = df.drop(columns=ts_cols)
        print(f"  [prep] Dropped timestamp columns: {ts_cols}")

    df = df.replace([np.inf, -np.inf], np.nan)
    n_before = len(df)
    df = df.dropna()
    n_dropped = n_before - len(df)
    print(f"  [prep] Dropped {n_dropped:,} rows with NaN/Inf  →  {len(df):,} remaining")
    return df.reset_index(drop=True)


def encode_labels(df: pd.DataFrame) -> tuple[pd.DataFrame, LabelEncoder]:
    """
    Encode string class labels to integers.

    Returns the DataFrame with an added 'Label_enc' column and the
    fitted LabelEncoder (needed to recover class names later).
    """
    label_col = _find_label_col(df)
    le = LabelEncoder()
    df = df.copy()
    df["Label_enc"] = le.fit_transform(df[label_col].astype(str).str.strip())
    class_counts = df["Label_enc"].value_counts().sort_index()
    print(f"  [prep] {len(le.classes_)} classes encoded:")
    for idx, name in enumerate(le.classes_):
        print(f"         {idx:2d}  {name}  ({class_counts.get(idx, 0):,} samples)")
    return df, le


def build_feature_matrix(
    df: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Extract numeric feature matrix X and integer label vector y.

    Non-numeric columns (other than the encoded label) are excluded.
    Returns feature names alongside arrays for downstream interpretability.
    """
    label_col = _find_label_col(df)
    exclude = {label_col, "Label_enc"}
    feature_cols = [
        c for c in df.columns
        if c not in exclude and pd.api.types.is_numeric_dtype(df[c])
    ]
    X = df[feature_cols].values.astype(np.float32)
    y = df["Label_enc"].values.astype(np.int32)
    print(f"  [prep] Feature matrix: {X.shape[0]:,} × {X.shape[1]}")
    return X, y, feature_cols


def split_data(
    X: np.ndarray,
    y: np.ndarray,
    test_size: float = 0.20,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Stratified 80/20 split — preserves class distribution in each fold."""
    return train_test_split(
        X, y,
        test_size=test_size,
        random_state=RANDOM_STATE,
        stratify=y,
    )


def scale_data(
    X_train: np.ndarray,
    X_test: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, StandardScaler]:
    """
    Fit StandardScaler on training data, apply to both splits.

    IMPORTANT: The scaler is fit only on X_train.  Fitting on the full
    dataset (or on X_test) would constitute data leakage.
    """
    scaler = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)
    return X_train_s, X_test_s, scaler
