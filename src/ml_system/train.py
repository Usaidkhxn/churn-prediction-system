from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    confusion_matrix,
)

from src.ml_system.config import load_config
from src.ml_system.data import load_churn_csv
from src.ml_system.features import build_preprocessor
from src.ml_system.model import build_model


def train_and_eval():
    cfg = load_config()

    X, y = load_churn_csv(cfg.data.path, cfg.data.target)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=cfg.data.test_size,
        random_state=cfg.random_state,
        stratify=y
    )

    preprocessor = build_preprocessor(X_train)
    model = build_model(cfg.train.model_type, cfg.random_state)

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("model", model),
    ])

    print("Training model...")
    pipe.fit(X_train, y_train)

    print("Evaluating...")
    y_pred = pipe.predict(X_test)

    # Probabilities for AUC metrics
    if hasattr(pipe, "predict_proba"):
        y_proba = pipe.predict_proba(X_test)[:, 1]
    else:
        # fallback: decision_function -> sigmoid-ish not guaranteed, but keeps flow
        scores = pipe.decision_function(X_test)
        y_proba = 1 / (1 + np.exp(-scores))

    metrics = {
        "model_type": cfg.train.model_type,
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "f1": float(f1_score(y_test, y_pred)),
        "precision": float(precision_score(y_test, y_pred)),
        "recall": float(recall_score(y_test, y_pred)),
        "roc_auc": float(roc_auc_score(y_test, y_proba)),
        "pr_auc": float(average_precision_score(y_test, y_proba)),
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
    }

    # Save artifacts
    Path(cfg.artifacts.dir).mkdir(parents=True, exist_ok=True)

    joblib.dump(pipe, cfg.artifacts.model_path)
    Path(cfg.artifacts.metrics_path).write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"Saved model: {cfg.artifacts.model_path}")
    print(f"Saved metrics: {cfg.artifacts.metrics_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    train_and_eval()
