"""
Store-level forecast page.
Generates 7-day ahead predictions for a selected store using the best trained model.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import (
    get_processed_df,
    load_scalers,
    MODELS_DIR,
    FEATURE_COLS,
    SEQ_LEN,
)
from inference import (
    load_trained_model,
    predict_store_next_n,
    load_training_summary,
    get_store_scalers,
)


@st.cache_data(ttl="1h", show_spinner="Processing features …")
def get_df():
    return get_processed_df()


@st.cache_resource(show_spinner="Loading model …")
def get_model(arch: str, n_features: int):
    return load_trained_model(arch, n_features)


def check_models_available() -> list[str]:
    available = []
    for arch in ["GRU", "LSTM", "RNN"]:
        if (MODELS_DIR / f"{arch.lower()}_best.pt").exists():
            available.append(arch)
    return available


st.title("Store-level demand forecast")
st.caption("Select a store and model to generate 7-day ahead inventory forecasts.")

available = check_models_available()

if not available:
    st.warning(
        "No trained models found. Please go to **Train models** first.",
        icon=":material/warning:",
    )
    st.stop()

# ── Sidebar controls ────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/tune: Forecast settings")

    arch = st.selectbox(
        "Model architecture",
        options=available,
        help="GRU achieved the best MAPE (~10.2%)",
    )
    store_id = st.number_input(
        "Store ID (1–1115)",
        min_value=1,
        max_value=1115,
        value=1,
        step=1,
    )
    forecast_horizon = st.slider(
        "Forecast horizon (days)",
        min_value=1,
        max_value=28,
        value=7,
    )
    show_history = st.slider(
        "Historical days to display",
        min_value=14,
        max_value=180,
        value=60,
    )

df = get_df()

if store_id not in df["Store"].values:
    st.error(f"Store {store_id} not found in processed data.")
    st.stop()

# Load scalers
try:
    all_scalers = load_scalers()
except FileNotFoundError:
    all_scalers = {}

scaler_X, scaler_y = get_store_scalers(df, store_id, all_scalers)
if store_id not in all_scalers:
    st.info(
        f"Store {store_id} features calibrated directly from its own historical sales distribution.",
        icon=":material/analytics:",
    )

# Load model
summary = load_training_summary()
n_features = summary["n_features"] if summary else len([c for c in FEATURE_COLS if c in df.columns])

try:
    model = get_model(arch, n_features)
except FileNotFoundError as e:
    st.error(str(e))
    st.stop()

# ── Historical actuals ──────────────────────────────────────────────────────
store_df = df[df["Store"] == store_id].sort_values("Date").reset_index(drop=True)
hist = store_df.tail(show_history)[["Date", "Sales", "Promo"]].copy()
hist["Type"] = "Actual"

# ── Forecast ────────────────────────────────────────────────────────────────
with st.spinner(f"Generating {forecast_horizon}-day forecast with {arch} …"):
    preds = predict_store_next_n(
        model, df, store_id, scaler_X, scaler_y, n_steps=forecast_horizon
    )

last_date = store_df["Date"].max()
future_dates = pd.date_range(last_date + pd.Timedelta(days=1), periods=forecast_horizon, freq="D")
forecast_df = pd.DataFrame({"Date": future_dates, "Sales": preds, "Type": "Forecast"})

# Combined for chart
combined = pd.concat([
    hist.rename(columns={"Sales": "Sales"}),
    forecast_df,
], ignore_index=True)

# ── Layout ───────────────────────────────────────────────────────────────────
st.subheader(f"Store {store_id} — historical sales & {forecast_horizon}-day forecast")

# KPI row
with st.container(horizontal=True):
    st.metric(
        "Avg actual sales (shown period)",
        f"€{hist['Sales'].mean():,.0f}",
        border=True,
    )
    st.metric(
        f"Avg {forecast_horizon}-day forecast",
        f"€{preds.mean():,.0f}",
        border=True,
    )
    st.metric(
        "Forecast max",
        f"€{preds.max():,.0f}",
        border=True,
    )
    promo_days_hist = hist["Promo"].sum() if "Promo" in hist.columns else 0
    st.metric(
        "Promo days in history",
        int(promo_days_hist),
        border=True,
    )

# Main chart
st.subheader("Sales chart")
col1, col2 = st.columns([3, 1])
with col1:
    # Plot actuals and forecast separately so colors differ
    chart_data = hist[["Date", "Sales"]].copy()
    chart_data = chart_data.rename(columns={"Sales": "Actual sales"})
    forecast_chart = forecast_df[["Date", "Sales"]].copy().rename(columns={"Sales": f"{arch} forecast"})

    # Merge on Date for chart
    full_chart = pd.merge(
        chart_data.set_index("Date"),
        forecast_chart.set_index("Date"),
        left_index=True,
        right_index=True,
        how="outer",
    ).reset_index().rename(columns={"index": "Date"})

    st.line_chart(full_chart, x="Date", y=["Actual sales", f"{arch} forecast"])

with col2:
    with st.container(border=True):
        st.markdown(f"**{forecast_horizon}-day forecast**")
        forecast_display = forecast_df[["Date", "Sales"]].copy()
        forecast_display["Date"] = forecast_display["Date"].dt.strftime("%Y-%m-%d")
        forecast_display["Sales"] = forecast_display["Sales"].apply(lambda x: f"€{x:,.0f}")
        st.dataframe(forecast_display, hide_index=True)

# ── Weekly breakdown ─────────────────────────────────────────────────────────
if forecast_horizon >= 7:
    st.subheader("Forecast by day of week pattern")
    hist_dow = (
        store_df.groupby(store_df["Date"].dt.dayofweek)["Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Date": "DoW", "Sales": "Avg historical"})
    )
    dow_map = {0: "Mon", 1: "Tue", 2: "Wed", 3: "Thu", 4: "Fri", 5: "Sat", 6: "Sun"}
    hist_dow["Day"] = hist_dow["DoW"].map(dow_map)
    st.bar_chart(hist_dow, x="Day", y="Avg historical", color="#60A5FA")

# ── Model info ────────────────────────────────────────────────────────────────
if summary:
    arch_results = next(
        (r for r in summary.get("results", []) if r["arch"] == arch), None
    )
    if arch_results:
        st.caption(
            f":material/analytics: {arch} validation MAPE: {arch_results['best_val_mape']:.2f}% "
            f"(trained across all {summary['config'].get('num_stores', 1115):,} stores)"
        )
