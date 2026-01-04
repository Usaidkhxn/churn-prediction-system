from __future__ import annotations

from fastapi import FastAPI
from pydantic import BaseModel, Field
import pandas as pd

from src.ml_system.config import load_config
from src.ml_system.data import load_churn_csv
from src.ml_system.predict import predict_one, score_dataframe
from src.ml_system.decision import select_targets

app = FastAPI(title="Churn Prediction System", version="1.0.0")
cfg = load_config()

# Infer expected features from the dataset columns (minus target)
X_demo, _ = load_churn_csv(cfg.data.path, cfg.data.target)
REQUIRED_FEATURES = list(X_demo.columns)


class PredictRequest(BaseModel):
    features: dict = Field(..., description="Feature dictionary for one customer (must include all required fields).")


class BatchPredictRequest(BaseModel):
    rows: list[dict] = Field(..., description="List of feature dictionaries for multiple customers.")


class TargetingRequest(BaseModel):
    rows: list[dict]
    budget: float = Field(..., description="Total budget available for retention offers.")


@app.get("/health")
def health():
    return {"status": "ok"}


@app.get("/schema")
def schema():
    return {"required_features": REQUIRED_FEATURES}


@app.post("/predict")
def predict(req: PredictRequest):
    missing = [f for f in REQUIRED_FEATURES if f not in req.features]
    if missing:
        return {"error": "Missing features", "missing": missing}
    return predict_one(req.features)


@app.post("/batch_predict")
def batch_predict(req: BatchPredictRequest):
    df = pd.DataFrame(req.rows)
    missing_cols = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_cols:
        return {"error": "Missing columns", "missing": missing_cols}

    df = df[REQUIRED_FEATURES]
    scored = score_dataframe(df)
    return scored[["churn_probability"]].to_dict(orient="records")


@app.post("/targeting")
def targeting(req: TargetingRequest):
    """
    Returns a ranked list of customers to target under the given budget using expected value.
    """
    df = pd.DataFrame(req.rows)
    missing_cols = [f for f in REQUIRED_FEATURES if f not in df.columns]
    if missing_cols:
        return {"error": "Missing columns", "missing": missing_cols}

    df = df[REQUIRED_FEATURES]
    scored = score_dataframe(df)

    b = cfg.business
    targets = select_targets(
        df_scored=scored,
        budget=req.budget,
        retention_offer_cost=float(b["retention_offer_cost"]),
        churn_value=float(b["churn_value"]),
        retention_success_prob=float(b["retention_success_prob"]),
        id_col=None,
    )

    return targets.to_dict(orient="records")
