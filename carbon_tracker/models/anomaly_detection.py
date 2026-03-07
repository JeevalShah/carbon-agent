from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest


def detect_carbon_hotspots(
    lane_df: pd.DataFrame,
    contamination: float = 0.08,
    random_state: int = 42,
) -> pd.DataFrame:
    """
    Detect abnormal lanes using IsolationForest.

    Features used (as requested):
      - total_emissions_per_lane
      - carbon_intensity

    Adds:
      - anomaly_score (higher means more anomalous after sign flip)
      - carbon_hotspot (True/False)
    """
    df = lane_df.copy()

    feature_cols = ["total_emissions_per_lane", "carbon_intensity"]
    X = df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(float)

    iso = IsolationForest(
        n_estimators=300,
        contamination=contamination,
        random_state=random_state,
    )
    preds = iso.fit_predict(X)  # -1 anomaly, 1 normal
    scores = iso.decision_function(X)  # higher is more normal

    df["anomaly_score"] = (-scores).astype(float)
    df["carbon_hotspot"] = preds == -1

    return df

