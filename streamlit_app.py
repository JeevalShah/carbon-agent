from __future__ import annotations

import pandas as pd
import plotly.express as px
import streamlit as st

from carbon_tracker.agents.genai_agent import answer_general_query, generate_lane_insight
from carbon_tracker.main import run_pipeline
from carbon_tracker.utils.optimization_engine import simulate_optimization


def _format_kg(value: float) -> str:
    return f"{value:,.0f} kg"


def _format_intensity(value: float) -> str:
    if value is None or pd.isna(value):
        return "NA"
    return f"{float(value):.4f}"


@st.cache_data(show_spinner=False, ttl=3600)
def _run_cached_pipeline_from_upload(upload_bytes: bytes | None):
    """
    Cache pipeline results per uploaded file contents (or default synthetic data).
    Added TTL of 3600 seconds (1 hour) to prevent stale cache issues.
    """
    if upload_bytes is None:
        out = run_pipeline(file_path=None)
        return out.shipments, out.lanes, out.emission_model

    # Streamlit provides bytes-like; pandas can read from BytesIO.
    from io import BytesIO

    bio = BytesIO(upload_bytes)
    df = pd.read_csv(bio)
    
    # Validate required columns for uploaded files
    required_cols = [
        "shipment_id", "origin", "destination", "distance_km",
        "vehicle_type", "fuel_type", "shipment_weight_tons", 
        "vehicle_capacity_tons", "date"
    ]
    missing_cols = [c for c in required_cols if c not in df.columns]
    if missing_cols:
        raise ValueError(
            f"Uploaded CSV is missing required columns: {missing_cols}. "
            f"Please upload a file with these columns: {required_cols}"
        )
    # Write to a temporary in-memory buffer isn't needed; run_pipeline expects a path,
    # so we run the same steps inline for uploaded data to avoid filesystem coupling.
    # Reuse run_pipeline behavior by saving a temporary file would be more complex on Windows.
    from carbon_tracker.models.anomaly_detection import detect_carbon_hotspots
    from carbon_tracker.models.emission_model import (
        compute_co2_emissions,
        ensure_fuel_consumption_column,
        predict_fuel_consumption,
        train_emission_model,
    )
    from carbon_tracker.utils.feature_engineering import preprocess_for_model
    from carbon_tracker.utils.lane_analytics import compute_lane_analytics

    feats = preprocess_for_model(df)
    feats = ensure_fuel_consumption_column(feats)
    artifacts = train_emission_model(feats)
    preds = predict_fuel_consumption(artifacts, feats)
    preds = compute_co2_emissions(preds, fuel_consumption_col="fuel_consumption_liters_pred", output_col="CO2_emissions_kg")
    lane_df = compute_lane_analytics(preds, emissions_col="CO2_emissions_kg")
    lane_df = detect_carbon_hotspots(lane_df)
    preds["lane"] = preds["origin"].astype(str) + "-" + preds["destination"].astype(str)
    return preds, lane_df, artifacts


def _emissions_by_lane_bar(lane_df: pd.DataFrame):
    fig = px.bar(
        lane_df.sort_values("total_emissions_per_lane", ascending=False).head(20),
        x="lane",
        y="total_emissions_per_lane",
        color="carbon_hotspot",
        title="Top 20 Lanes by Total Emissions",
        labels={"total_emissions_per_lane": "Total CO2 (kg)"},
    )
    fig.update_layout(xaxis_tickangle=-45, height=450)
    return fig


def _carbon_intensity_heatmap(lane_df: pd.DataFrame):

    if lane_df.empty:
        return px.imshow([[0]], title="No data available")

    pivot = lane_df.pivot_table(
        index="origin",
        columns="destination",
        values="carbon_intensity",
        aggfunc="mean",
        fill_value=0
    )

    if pivot.shape[0] == 0 or pivot.shape[1] == 0:
        return px.imshow([[0]], title="Insufficient data")

    pivot = pd.DataFrame(pivot)

    fig = px.imshow(
        pivot,
        aspect="auto",
        color_continuous_scale="Reds",
        title="Carbon Intensity Heatmap (kg CO2 per tonne-km)",
        labels=dict(color="kg CO2/tonne-km")
    )

    fig.update_layout(
        height=450,
        xaxis_title="Destination",
        yaxis_title="Origin"
    )

    fig.update_traces(
        hovertemplate="Origin: %{y}<br>Destination: %{x}<br>Intensity: %{z:.4f}"
    )

    return fig

def _hotspot_table(lane_df: pd.DataFrame):
    hs = lane_df[lane_df["carbon_hotspot"] == True].copy()  # noqa: E712
    hs = hs.sort_values(["anomaly_score", "total_emissions_per_lane"], ascending=False)
    cols = [
        "lane",
        "total_emissions_per_lane",
        "carbon_intensity",
        "avg_load_factor",
        "avg_distance_km",
        "shipment_count",
        "anomaly_score",
    ]
    cols = [c for c in cols if c in hs.columns]
    return hs[cols]


def main():
    st.set_page_config(page_title="Carbon Tracker Agent", layout="wide")
    st.title("Carbon Tracker Agent for Road Freight Logistics")
    st.caption("Estimate CO2 emissions, detect hotspots, simulate optimizations, and ask a GenAI assistant.")

    with st.sidebar:
        st.header("Data")
        uploaded = st.file_uploader("Upload shipment CSV", type=["csv"])
        st.write("If you don't upload a file, the app uses synthetic data.")

        st.header("GenAI")
        st.write("Set `OPENAI_API_KEY` in your environment to enable AI insights.")

    upload_bytes = uploaded.getvalue() if uploaded is not None else None
    
    try:
        shipments_df, lane_df, model_artifacts = _run_cached_pipeline_from_upload(upload_bytes)
    except ValueError as e:
        st.error(str(e))
        if uploaded is not None:
            st.info("Please upload a CSV file with the required columns, or leave empty to use default data.")
        return

    total_emissions = float(shipments_df["CO2_emissions_kg"].sum())
    avg_intensity = float((lane_df["carbon_intensity"].replace([pd.NA], 0).fillna(0)).mean())
    total_shipments = int(shipments_df.shape[0])

    c1, c2, c3 = st.columns(3)
    c1.metric("Total emissions", _format_kg(total_emissions))
    c2.metric("Average carbon intensity", _format_intensity(avg_intensity))
    c3.metric("Total shipments", f"{total_shipments:,}")

    st.divider()

    st.subheader("Lane Analytics")
    st.dataframe(
        lane_df.sort_values("total_emissions_per_lane", ascending=False),
        use_container_width=True,
        hide_index=True,
    )

    left, right = st.columns(2)
    with left:
        st.plotly_chart(_emissions_by_lane_bar(lane_df), use_container_width=True)
    with right:
        fig = _carbon_intensity_heatmap(lane_df)
        st.plotly_chart(fig, use_container_width=True)

    st.write(lane_df.shape)
    st.write(lane_df.head())

    st.write("Carbon intensity stats")
    st.write(lane_df["carbon_intensity"].describe())

    st.subheader("Carbon Hotspot Alerts")
    hs = _hotspot_table(lane_df)
    if hs.empty:
        st.success("No carbon hotspots detected with current settings.")
    else:
        st.warning(f"Detected {hs.shape[0]} hotspot lanes.")
        st.dataframe(hs, use_container_width=True, hide_index=True)

    st.divider()
    st.subheader("Optimization Simulator")

    scenario = st.selectbox(
        "Choose a scenario",
        [
            "Increase load factor to 0.75",
            "Improve route distance by 10%",
            "Switch to more efficient vehicle",
        ],
    )
    if st.button("Simulate optimization"):
        res = simulate_optimization(shipments_df, scenario=scenario, model_artifacts=model_artifacts)
        st.write(
            {
                "baseline_emissions_kg": round(res.baseline_emissions_kg, 2),
                "optimized_emissions_kg": round(res.optimized_emissions_kg, 2),
                "percentage_reduction": round(res.percentage_reduction, 2),
            }
        )
        fig = px.bar(
            pd.DataFrame(
                [
                    {"case": "Baseline", "emissions_kg": res.baseline_emissions_kg},
                    {"case": "Optimized", "emissions_kg": res.optimized_emissions_kg},
                ]
            ),
            x="case",
            y="emissions_kg",
            title="Baseline vs Optimized Emissions",
            labels={"emissions_kg": "CO2 (kg)"},
        )
        st.plotly_chart(fig, use_container_width=True)

    st.divider()
    st.subheader("GenAI Assistant Chat")

    if "chat_messages" not in st.session_state:
        st.session_state.chat_messages = []

    for m in st.session_state.chat_messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    user_q = st.chat_input("Ask about emissions, hotspots, or recommendations...")
    if user_q:
        st.session_state.chat_messages.append({"role": "user", "content": user_q})
        with st.chat_message("user"):
            st.markdown(user_q)

        # Attempt grounded answer first; fallback is automatic.
        resp = answer_general_query(user_q, lane_df=lane_df)

        with st.chat_message("assistant"):
            st.markdown(resp.answer)

        st.session_state.chat_messages.append({"role": "assistant", "content": resp.answer})

    st.divider()
    st.subheader("Lane Deep-Dive (AI Insight)")

    lane_choice = st.selectbox("Pick a lane", lane_df["lane"].astype(str).tolist())
    lane_row = lane_df[lane_df["lane"].astype(str) == lane_choice].iloc[0]
    if st.button("Generate lane insight"):
        resp = generate_lane_insight(lane_row, user_query=f"Why is {lane_choice} a high emission lane?")
        st.markdown(resp.answer)


if __name__ == "__main__":
    main()

