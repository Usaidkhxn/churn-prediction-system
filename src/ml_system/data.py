from __future__ import annotations

from pathlib import Path
import pandas as pd


def _normalize_churn_target(y: pd.Series) -> pd.Series:
    """
    Normalize churn target to 0/1.
    Supports: Yes/No, TRUE/FALSE, 1/0, churned/not churned variants.
    """
    # If already numeric 0/1
    if pd.api.types.is_numeric_dtype(y):
        # coerce to int 0/1 if possible
        y2 = pd.to_numeric(y, errors="coerce")
        if set(y2.dropna().unique()).issubset({0, 1}):
            return y2.fillna(0).astype(int)

    y_str = y.astype(str).str.strip().str.lower()

    mapping = {
        "yes": 1, "y": 1, "true": 1, "1": 1, "churn": 1, "churned": 1,
        "no": 0, "n": 0, "false": 0, "0": 0, "not churn": 0, "not churned": 0
    }
    y_mapped = y_str.map(mapping)

    if y_mapped.isna().any():
        bad = sorted(set(y_str[y_mapped.isna()].unique().tolist()))
        raise ValueError(
            f"Target column has unexpected values that cannot be mapped to 0/1: {bad[:20]}"
            + (" ..." if len(bad) > 20 else "")
        )

    return y_mapped.astype(int)


def load_churn_csv(path: str, target_col: str) -> tuple[pd.DataFrame, pd.Series]:
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Dataset not found at: {p.resolve()}")

    df = pd.read_csv(p)

    # Basic sanity checks
    if df.shape[0] == 0:
        raise ValueError("Dataset has 0 rows.")
    if target_col not in df.columns:
        raise ValueError(f"Target column '{target_col}' not found. Columns: {list(df.columns)[:30]}")

    # Drop completely empty columns
    df = df.dropna(axis=1, how="all")

    # Remove duplicate columns if any (rare but happens)
    if df.columns.duplicated().any():
        df = df.loc[:, ~df.columns.duplicated()].copy()

    y_raw = df[target_col]
    X = df.drop(columns=[target_col]).copy()

    # Common cleanup: if dataset has an ID column
    for id_col in ["customerID", "CustomerID", "id", "ID"]:
        if id_col in X.columns:
            X = X.drop(columns=[id_col])

    y = _normalize_churn_target(y_raw)

    # Remove rows where all features are NA
    all_na = X.isna().all(axis=1)
    if all_na.any():
        X = X.loc[~all_na].copy()
        y = y.loc[~all_na].copy()

    return X, y
