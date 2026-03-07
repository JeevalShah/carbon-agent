from __future__ import annotations

import pandas as pd


VEHICLE_EFFICIENCY_FACTOR = {
    "truck_small": 1.2,
    "truck_medium": 1.0,
    "truck_large": 0.85,
}

VEHICLE_TYPE_ENCODING = {
    "truck_small": 0,
    "truck_medium": 1,
    "truck_large": 2,
}

FUEL_TYPE_ENCODING = {"diesel": 0, "petrol": 1}


def add_logistics_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute core logistics features used throughout the pipeline.

    - load_factor = shipment_weight_tons / vehicle_capacity_tons
    - tonne_km = shipment_weight_tons * distance_km
    - vehicle_efficiency_factor based on vehicle_type
    """
    out = df.copy()

    out["load_factor"] = out["shipment_weight_tons"] / out["vehicle_capacity_tons"]
    out["load_factor"] = out["load_factor"].clip(lower=0.01, upper=1.5)

    out["tonne_km"] = out["shipment_weight_tons"] * out["distance_km"]
    out["tonne_km"] = out["tonne_km"].clip(lower=0.0)

    out["vehicle_efficiency_factor"] = out["vehicle_type"].map(VEHICLE_EFFICIENCY_FACTOR).fillna(1.0)

    return out


def encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """
    Encode categorical variables for ML.

    Creates:
      - vehicle_type_encoded
      - fuel_type_encoded
    """
    out = df.copy()

    out["vehicle_type_encoded"] = out["vehicle_type"].map(VEHICLE_TYPE_ENCODING).fillna(-1).astype(int)
    out["fuel_type_encoded"] = out["fuel_type"].map(FUEL_TYPE_ENCODING).fillna(-1).astype(int)

    return out


def preprocess_for_model(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full feature engineering pipeline: add logistics features then encode categoricals.
    """
    out = add_logistics_features(df)
    out = encode_categoricals(out)
    return out

