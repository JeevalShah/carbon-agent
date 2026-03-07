from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

import pandas as pd

from carbon_tracker.models.anomaly_detection import detect_carbon_hotspots
from carbon_tracker.models.emission_model import (
    EmissionModelArtifacts,
    compute_co2_emissions,
    ensure_fuel_consumption_column,
    predict_fuel_consumption,
    train_emission_model,
)
from carbon_tracker.utils.data_loader import load_shipment_data
from carbon_tracker.utils.feature_engineering import preprocess_for_model
from carbon_tracker.utils.lane_analytics import compute_lane_analytics


@dataclass
class PipelineOutput:
    shipments: pd.DataFrame
    lanes: pd.DataFrame
    emission_model: EmissionModelArtifacts


def run_pipeline(file_path: Optional[str] = None) -> PipelineOutput:
    """
    End-to-end pipeline for the Carbon Tracker Agent.

    Steps:
      - Load dataset (or generate synthetic)
      - Feature engineering
      - Ensure fuel consumption target exists (simulate if missing)
      - Train RandomForestRegressor to estimate fuel consumption
      - Predict fuel consumption and compute CO2 emissions
      - Compute lane-level analytics
      - Detect carbon hotspots (Isolation Forest)
    """
    raw = load_shipment_data(file_path)

    feats = preprocess_for_model(raw)
    feats = ensure_fuel_consumption_column(feats)

    artifacts = train_emission_model(feats)
    preds = predict_fuel_consumption(artifacts, feats)
    preds = compute_co2_emissions(preds, fuel_consumption_col="fuel_consumption_liters_pred", output_col="CO2_emissions_kg")

    lane_df = compute_lane_analytics(preds, emissions_col="CO2_emissions_kg")
    lane_df = detect_carbon_hotspots(lane_df)

    # Add lane string to shipment-level df for dashboard linking.
    preds["lane"] = preds["origin"].astype(str) + "-" + preds["destination"].astype(str)

    return PipelineOutput(shipments=preds, lanes=lane_df, emission_model=artifacts)


def main() -> None:
    """
    CLI entry point for quick local testing.
    """
    out = run_pipeline()
    print("Shipments:", out.shipments.shape)
    print("Lanes:", out.lanes.shape)
    print(out.lanes.head(10)[["lane", "total_emissions_per_lane", "carbon_intensity", "carbon_hotspot"]])


if __name__ == "__main__":
    main()

