from __future__ import annotations

import pandas as pd
import numpy as np


def compute_expected_value(
    churn_proba: np.ndarray,
    churn_value: float,
    retention_success_prob: float,
    retention_offer_cost: float,
) -> np.ndarray:
    """
    Expected value of targeting a customer with a retention offer.

    EV = P(churn) * P(offer works) * value_saved  -  cost_of_offer
    """
    return (churn_proba * retention_success_prob * churn_value) - retention_offer_cost


def select_targets(
    df_scored: pd.DataFrame,
    budget: float,
    retention_offer_cost: float,
    churn_value: float,
    retention_success_prob: float,
    id_col: str | None = None,
) -> pd.DataFrame:
    """
    Takes a dataframe that includes 'churn_probability' and returns the best targets under a given budget.
    """
    if "churn_probability" not in df_scored.columns:
        raise ValueError("df_scored must contain a 'churn_probability' column")

    churn_proba = df_scored["churn_probability"].to_numpy()

    ev = compute_expected_value(
        churn_proba=churn_proba,
        churn_value=churn_value,
        retention_success_prob=retention_success_prob,
        retention_offer_cost=retention_offer_cost,
    )

    out = df_scored.copy()
    out["expected_value"] = ev

    # How many can we afford?
    max_targets = int(budget // retention_offer_cost)
    max_targets = max(max_targets, 0)

    # Sort by expected value descending and keep only positive EV
    out = out.sort_values("expected_value", ascending=False)
    out = out[out["expected_value"] > 0].head(max_targets)

    # Optional: keep only id + proba + ev for clean output
    cols = []
    if id_col and id_col in out.columns:
        cols.append(id_col)
    cols += ["churn_probability", "expected_value"]

    return out[cols] if cols else out
