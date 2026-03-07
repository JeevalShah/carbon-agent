from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor


EMISSION_FACTORS_KG_CO2_PER_LITER = {
    "diesel": 2.68,
    "petrol": 2.31,
}


DEFAULT_FEATURE_COLUMNS = [
    "distance_km",
    "shipment_weight_tons",
    "load_factor",
    "tonne_km",
    "vehicle_type_encoded",
]


def simulate_fuel_consumption_liters(df: pd.DataFrame) -> pd.Series:
    """
    Simulate fuel consumption when the dataset doesn't contain it.

    Heuristic formula (as requested):
      fuel_consumption_liters = (distance_km / 4) * (1 + load_factor * 0.3)

    Adds a small amount of noise so the model has a signal to learn beyond a perfect rule.
    """
    base = (df["distance_km"] / 4.0) * (1.0 + df["load_factor"] * 0.3)

    # Add mild multiplicative noise (±5%) to mimic real-world variability.
    rng = np.random.default_rng(42)
    noise = rng.normal(loc=1.0, scale=0.05, size=len(df))
    return (base * noise).clip(lower=0.1)


def ensure_fuel_consumption_column(df: pd.DataFrame) -> pd.DataFrame:
    """
    Ensure df has 'fuel_consumption_liters'. If missing, create it via simulation.
    """
    out = df.copy()
    if "fuel_consumption_liters" not in out.columns:
        out["fuel_consumption_liters"] = simulate_fuel_consumption_liters(out)
    else:
        out["fuel_consumption_liters"] = pd.to_numeric(out["fuel_consumption_liters"], errors="coerce")
        missing = out["fuel_consumption_liters"].isna()
        if missing.any():
            out.loc[missing, "fuel_consumption_liters"] = simulate_fuel_consumption_liters(out.loc[missing])
    return out


@dataclass
class EmissionModelArtifacts:
    """
    Bundle the trained model and the exact feature column order used during training.
    """

    model: RandomForestRegressor
    feature_columns: List[str]


def train_emission_model(
    df: pd.DataFrame,
    feature_columns: Optional[List[str]] = None,
    random_state: int = 42,
) -> EmissionModelArtifacts:
    """
    Train a RandomForestRegressor to estimate fuel consumption (liters).
    """
    cols = feature_columns or DEFAULT_FEATURE_COLUMNS
    X = df[cols].copy()
    y = df["fuel_consumption_liters"].astype(float).copy()

    model = RandomForestRegressor(
        n_estimators=300,
        random_state=random_state,
        n_jobs=-1,
        min_samples_leaf=2,
    )
    model.fit(X, y)
    return EmissionModelArtifacts(model=model, feature_columns=cols)


def predict_fuel_consumption(
    artifacts: EmissionModelArtifacts,
    df: pd.DataFrame,
    output_col: str = "fuel_consumption_liters_pred",
) -> pd.DataFrame:
    """
    Predict fuel consumption and append it as a column.
    """
    out = df.copy()
    X = out[artifacts.feature_columns].copy()
    preds = artifacts.model.predict(X)
    out[output_col] = np.clip(preds, 0.0, None)
    return out


def compute_co2_emissions(
    df: pd.DataFrame,
    fuel_consumption_col: str = "fuel_consumption_liters_pred",
    output_col: str = "CO2_emissions_kg",
) -> pd.DataFrame:
    """
    Compute CO2 emissions using fuel-type-specific emission factors.

    diesel = 2.68 kg CO2/liter
    petrol = 2.31 kg CO2/liter
    """
    out = df.copy()
    factors = out["fuel_type"].map(EMISSION_FACTORS_KG_CO2_PER_LITER).fillna(2.5)
    out[output_col] = pd.to_numeric(out[fuel_consumption_col], errors="coerce").fillna(0.0) * factors
    return out

