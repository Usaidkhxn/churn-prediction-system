from __future__ import annotations

import pandas as pd
from sklearn.compose import ColumnTransformer, make_column_selector as selector
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.impute import SimpleImputer


def build_preprocessor(X: pd.DataFrame) -> ColumnTransformer:
    """
    Builds a robust preprocessor for mixed-type churn data:
    - Numeric: median impute + scale
    - Categorical: most_frequent impute + one-hot
    """
    numeric_selector = selector(dtype_include=["int64", "float64", "int32", "float32"])
    cat_selector = selector(dtype_include=["object", "bool", "category"])

    numeric_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler())
    ])

    categorical_pipeline = Pipeline(steps=[
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=True))
    ])

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, numeric_selector),
            ("cat", categorical_pipeline, cat_selector),
        ],
        remainder="drop",
        verbose_feature_names_out=False
    )

    # Fit sanity check happens when pipeline fits
    return preprocessor
