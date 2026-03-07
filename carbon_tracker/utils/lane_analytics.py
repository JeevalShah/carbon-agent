from __future__ import annotations

import pandas as pd


def compute_lane_analytics(
    shipment_df: pd.DataFrame,
    emissions_col: str = "CO2_emissions_kg",
) -> pd.DataFrame:
    """
    Aggregate shipment data at lane level.

    Lane definition: origin + "-" + destination

    Outputs include:
      - total_emissions_per_lane
      - total_tonne_km
      - average_carbon_intensity (kg CO2 / tonne-km)
      - rankings by total emissions and carbon intensity
    """
    df = shipment_df.copy()
    df["lane"] = df["origin"].astype(str) + "-" + df["destination"].astype(str)

    # Carbon intensity per shipment (avoid divide-by-zero).
    denom = df["tonne_km"].replace(0, pd.NA)
    df["carbon_intensity"] = (df[emissions_col] / denom).astype("float64")

    grouped = df.groupby("lane", as_index=False).agg(
        total_emissions_per_lane=(emissions_col, "sum"),
        total_tonne_km=("tonne_km", "sum"),
        avg_carbon_intensity=("carbon_intensity", "mean"),
        avg_load_factor=("load_factor", "mean"),
        avg_distance_km=("distance_km", "mean"),
        shipment_count=("shipment_id", "count"),
        origin=("origin", "first"),
        destination=("destination", "first"),
    )

    # Recompute intensity at lane level using totals (more stable than mean of ratios).
    grouped["carbon_intensity"] = (
        grouped["total_emissions_per_lane"] / grouped["total_tonne_km"].replace(0, pd.NA)
    ).astype("float64")

    grouped["rank_total_emissions"] = grouped["total_emissions_per_lane"].rank(
        ascending=False, method="dense"
    ).astype(int)
    grouped["rank_carbon_intensity"] = grouped["carbon_intensity"].rank(
        ascending=False, method="dense"
    ).astype(int)

    grouped = grouped.sort_values(["total_emissions_per_lane"], ascending=False).reset_index(drop=True)
    return grouped

