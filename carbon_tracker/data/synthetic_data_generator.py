from __future__ import annotations

import random
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd


ORIGINS = ["Mumbai", "Delhi", "Bangalore", "Chennai", "Hyderabad", "Pune", "Ahmedabad"]
DESTINATIONS = ORIGINS.copy()
VEHICLE_TYPES = ["truck_small", "truck_medium", "truck_large"]
FUEL_TYPES = ["diesel", "petrol"]


def _random_date(start: datetime, end: datetime, rng: random.Random) -> str:
    """Return a random date (YYYY-MM-DD) between start and end."""
    delta_days = (end - start).days
    d = start + timedelta(days=rng.randint(0, max(delta_days, 0)))
    return d.strftime("%Y-%m-%d")


def generate_synthetic_shipments(
    n: int = 1000,
    seed: int = 42,
    output_csv_path: Optional[str | Path] = None,
) -> pd.DataFrame:
    """
    Generate a synthetic road freight shipment dataset with realistic ranges.

    Columns:
      shipment_id, origin, destination, distance_km, vehicle_type, fuel_type,
      shipment_weight_tons, vehicle_capacity_tons, date

    If output_csv_path is provided, saves the CSV to that location.
    """
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    # Typical India road distances are skewed; use a mixture to avoid uniform-looking data.
    # Keep values within 50-2000 km as required.
    distances = np.clip(
        np_rng.lognormal(mean=6.2, sigma=0.55, size=n),  # produces a long tail
        50,
        2000,
    ).round(0)

    vehicle_types = np_rng.choice(VEHICLE_TYPES, size=n, p=[0.35, 0.45, 0.20]).tolist()
    fuel_types = np_rng.choice(FUEL_TYPES, size=n, p=[0.85, 0.15]).tolist()

    # Capacity depends on vehicle type (still within 10-30 tons).
    capacities = []
    for vt in vehicle_types:
        if vt == "truck_small":
            cap = rng.uniform(10, 16)
        elif vt == "truck_medium":
            cap = rng.uniform(14, 24)
        else:
            cap = rng.uniform(20, 30)
        capacities.append(round(cap, 2))

    # Shipment weights are constrained by capacity and the requested 2-25 range.
    weights = []
    for cap in capacities:
        # Most shipments are partially loaded; sample a load factor then multiply by capacity.
        lf = float(np.clip(np_rng.normal(loc=0.58, scale=0.18), 0.15, 0.98))
        w = min(cap * lf, 25.0)
        w = max(w, 2.0)
        weights.append(round(w, 2))

    origins = np_rng.choice(ORIGINS, size=n).tolist()
    destinations = []
    for o in origins:
        # Allow same-city lanes occasionally, but prefer different cities.
        if rng.random() < 0.90:
            d = rng.choice([x for x in DESTINATIONS if x != o])
        else:
            d = o
        destinations.append(d)

    start = datetime.now() - timedelta(days=365)
    end = datetime.now()
    dates = [_random_date(start, end, rng) for _ in range(n)]

    df = pd.DataFrame(
        {
            "shipment_id": [f"SHP-{i+1:05d}" for i in range(n)],
            "origin": origins,
            "destination": destinations,
            "distance_km": distances.astype(int),
            "vehicle_type": vehicle_types,
            "fuel_type": fuel_types,
            "shipment_weight_tons": weights,
            "vehicle_capacity_tons": capacities,
            "date": dates,
        }
    )

    if output_csv_path is not None:
        out_path = Path(output_csv_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out_path, index=False)

    return df


def generate_default_dataset() -> pd.DataFrame:
    """
    Generate and save the default dataset to carbon_tracker/data/shipment_data.csv.
    """
    default_path = Path(__file__).resolve().parent / "shipment_data.csv"
    return generate_synthetic_shipments(n=1000, seed=42, output_csv_path=default_path)

