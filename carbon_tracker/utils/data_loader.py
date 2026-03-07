from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from carbon_tracker.data.synthetic_data_generator import generate_synthetic_shipments


REQUIRED_COLUMNS = [
    "shipment_id",
    "origin",
    "destination",
    "distance_km",
    "vehicle_type",
    "fuel_type",
    "shipment_weight_tons",
    "vehicle_capacity_tons",
    "date",
]


def _is_missing_or_empty_csv(path: Path) -> bool:
    if not path.exists():
        return True
    try:
        if path.stat().st_size < 50:
            return True
    except OSError:
        return True
    try:
        df = pd.read_csv(path)
        return df.shape[0] == 0
    except Exception:
        return True


def load_shipment_data(file_path: Optional[str | Path] = None) -> pd.DataFrame:
    """
    Load shipment data from a CSV file.

    If file_path is None, uses the default path: carbon_tracker/data/shipment_data.csv.
    If the file is missing or empty, generates a synthetic dataset and saves it.
    """
    if file_path is None:
        csv_path = Path(__file__).resolve().parents[1] / "data" / "shipment_data.csv"
    else:
        csv_path = Path(file_path)

    if _is_missing_or_empty_csv(csv_path):
        df = generate_synthetic_shipments(n=1000, seed=42, output_csv_path=csv_path)
        return df

    df = pd.read_csv(csv_path)

    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(
            f"Dataset is missing required columns: {missing}. "
            f"Expected at least: {REQUIRED_COLUMNS}"
        )

    # Normalize types.
    numeric_cols = ["distance_km", "shipment_weight_tons", "vehicle_capacity_tons"]
    for c in numeric_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype(str)
    df = df.dropna(subset=numeric_cols + ["origin", "destination", "vehicle_type", "fuel_type"])

    return df

