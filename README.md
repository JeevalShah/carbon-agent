# Carbon Tracker Agent for Road Freight Logistics

Production-style prototype that estimates CO2 emissions for road freight shipments, provides lane-level sustainability analytics, flags carbon hotspots via anomaly detection, and offers a GenAI assistant for insights.

## Architecture

- **Data**: `carbon_tracker/data/synthetic_data_generator.py` generates a realistic synthetic dataset and saves it to `carbon_tracker/data/shipment_data.csv` if missing.
- **Ingestion**: `carbon_tracker/utils/data_loader.py` loads the dataset (or generates it if missing/empty).
- **Feature Engineering**: `carbon_tracker/utils/feature_engineering.py` computes:
  - load factor
  - tonne-km
  - vehicle efficiency factor
  - categorical encodings
- **Emission ML**: `carbon_tracker/models/emission_model.py`
  - Simulates `fuel_consumption_liters` if missing using a heuristic formula
  - Trains a `RandomForestRegressor` to predict fuel consumption
  - Calculates CO2 emissions using fuel-type emission factors
- **Lane Analytics**: `carbon_tracker/utils/lane_analytics.py` aggregates emissions and carbon intensity by lane.
- **Anomaly Detection**: `carbon_tracker/models/anomaly_detection.py` uses `IsolationForest` to flag carbon hotspots.
- **Optimization Simulator**: `carbon_tracker/utils/optimization_engine.py` simulates operational improvements and estimates emission reduction.
- **GenAI Agent**: `carbon_tracker/agents/genai_agent.py` uses OpenAI or Gemini (with a safe fallback if no key).
- **Dashboard**: `carbon_tracker/dashboard/streamlit_app.py` provides an interactive Streamlit + Plotly UI with uploads, charts, hotspots, optimization simulation, and chat.

## Setup

```bash
pip install -r requirements.txt
```

### (Optional) Enable GenAI

- **Gemini (Google) (optional)** — PowerShell:

```bash
$env:GOOGLE_API_KEY="YOUR_GEMINI_KEY_HERE"
```

`GOOGLE_API_KEY` and `GEMINI_API_KEY` are both supported for Gemini.

## Run

Launch the dashboard:

```bash
streamlit run carbon_tracker/streamlit_app.py
```

Run the pipeline from CLI (quick sanity check):

```bash
python -m carbon_tracker.main
```

