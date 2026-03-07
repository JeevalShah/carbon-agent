from __future__ import annotations

from dataclasses import dataclass
from typing import Literal, Optional

import numpy as np
import pandas as pd

from carbon_tracker.models.emission_model import (
    EMISSION_FACTORS_KG_CO2_PER_LITER,
    EmissionModelArtifacts,
    simulate_fuel_consumption_liters,
)
from carbon_tracker.utils.feature_engineering import preprocess_for_model


Scenario = Literal[
    "Increase load factor to 0.75",
    "Improve route distance by 10%",
    "Switch to more efficient vehicle",
]


@dataclass
class OptimizationResult:
    baseline_emissions_kg: float
    optimized_emissions_kg: float
    percentage_reduction: float


def _compute_emissions_from_fuel(df: pd.DataFrame, fuel_col: str) -> pd.Series:
    factors = df["fuel_type"].map(EMISSION_FACTORS_KG_CO2_PER_LITER).fillna(2.5)
    return pd.to_numeric(df[fuel_col], errors="coerce").fillna(0.0) * factors


def _switch_to_more_efficient_vehicle(vehicle_type: str) -> str:
    """
    Simulate an operational upgrade to a more efficient vehicle.
    """
    if vehicle_type == "truck_small":
        return "truck_medium"
    if vehicle_type == "truck_medium":
        return "truck_large"
    return "truck_large"


def simulate_optimization(
    shipment_df: pd.DataFrame,
    scenario: Scenario,
    model_artifacts: Optional[EmissionModelArtifacts] = None,
) -> OptimizationResult:
    """
    Simulate operational improvements and estimate emissions reduction.

    Scenarios:
      1) Increase load factor to 0.75
      2) Improve route distance by 10%
      3) Switch to more efficient vehicle

    If model_artifacts is provided, uses the trained RandomForestRegressor to estimate fuel.
    Otherwise falls back to the heuristic fuel simulation formula.
    """
    baseline = shipment_df.copy()
    if "fuel_consumption_liters_pred" in baseline.columns:
        baseline_fuel_col = "fuel_consumption_liters_pred"
    elif "fuel_consumption_liters" in baseline.columns:
        baseline_fuel_col = "fuel_consumption_liters"
    else:
        baseline_fuel_col = "fuel_consumption_liters_sim"
        baseline[baseline_fuel_col] = simulate_fuel_consumption_liters(baseline)

    baseline_emissions = float(_compute_emissions_from_fuel(baseline, baseline_fuel_col).sum())

    optimized = shipment_df.copy()

    if scenario == "Increase load factor to 0.75":
        # Assume better consolidation increases effective shipped weight up to 75% of capacity.
        current_lf = optimized["shipment_weight_tons"] / optimized["vehicle_capacity_tons"]
        target_weight = optimized["vehicle_capacity_tons"] * 0.75
        optimized["shipment_weight_tons"] = np.where(current_lf < 0.75, target_weight, optimized["shipment_weight_tons"])

    elif scenario == "Improve route distance by 10%":
        optimized["distance_km"] = (optimized["distance_km"] * 0.90).clip(lower=1.0)

    elif scenario == "Switch to more efficient vehicle":
        optimized["vehicle_type"] = optimized["vehicle_type"].astype(str).map(_switch_to_more_efficient_vehicle)
        # Keep capacity in range; assume upgraded vehicles typically have slightly higher capacity.
        optimized["vehicle_capacity_tons"] = (optimized["vehicle_capacity_tons"] * 1.10).clip(lower=10.0, upper=30.0)
        # Ensure weight does not exceed capacity after the change.
        optimized["shipment_weight_tons"] = np.minimum(optimized["shipment_weight_tons"], optimized["vehicle_capacity_tons"])

    else:
        raise ValueError(f"Unknown scenario: {scenario}")

    optimized = preprocess_for_model(optimized)

    if model_artifacts is not None:
        X = optimized[model_artifacts.feature_columns]
        optimized["fuel_consumption_liters_opt"] = np.clip(model_artifacts.model.predict(X), 0.0, None)
    else:
        optimized["fuel_consumption_liters_opt"] = simulate_fuel_consumption_liters(optimized)

    optimized_emissions = float(_compute_emissions_from_fuel(optimized, "fuel_consumption_liters_opt").sum())

    pct_reduction = 0.0 if baseline_emissions <= 0 else (baseline_emissions - optimized_emissions) / baseline_emissions * 100.0

    return OptimizationResult(
        baseline_emissions_kg=baseline_emissions,
        optimized_emissions_kg=optimized_emissions,
        percentage_reduction=float(pct_reduction),
    )

