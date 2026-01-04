from __future__ import annotations

from pathlib import Path
import joblib
import pandas as pd

from src.ml_system.config import load_config


def load_pipeline():
    cfg = load_config()
    model_path = Path(cfg.artifacts.model_path)
    if not model_path.exists():
        raise FileNotFoundError(
            f"Model not found at {model_path.resolve()}. Run training first: python -m src.ml_system.train"
        )
    return joblib.load(model_path)


def score_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    pipe = load_pipeline()
    proba = pipe.predict_proba(df)[:, 1]
    out = df.copy()
    out["churn_probability"] = proba
    return out


def predict_one(features: dict) -> dict:
    df = pd.DataFrame([features])
    scored = score_dataframe(df)
    p = float(scored["churn_probability"].iloc[0])
    pred = int(p >= 0.5)
    return {"prediction": pred, "churn_probability": p}
