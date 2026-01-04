from __future__ import annotations

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier


def build_model(model_type: str, random_state: int):
    if model_type == "logreg":
        return LogisticRegression(
            max_iter=4000,
            class_weight="balanced",  # churn is imbalanced
            solver="lbfgs"
        )

    if model_type == "random_forest":
        return RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            n_jobs=-1,
            class_weight="balanced_subsample"
        )

    raise ValueError(f"Unknown model_type: {model_type}")
