"""
Overview / Home page.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import load_raw_data, merge_and_clean, MODELS_DIR


@st.cache_data(ttl="1h", show_spinner="Loading Rossmann dataset …")
def get_summary_stats() -> dict:
    train, store = load_raw_data()
    df = merge_and_clean(train, store)
    open_df = df  # already filtered to open stores

    return {
        "total_records": len(train),
        "open_records": len(open_df),
        "n_stores": open_df["Store"].nunique(),
        "date_min": str(open_df["Date"].min().date()),
        "date_max": str(open_df["Date"].max().date()),
        "avg_sales": open_df["Sales"].mean(),
        "total_revenue": open_df["Sales"].sum(),
        "promo_pct": open_df["Promo"].mean() * 100,
        "avg_customers": open_df["Customers"].mean(),
        "store_types": store["StoreType"].value_counts().to_dict(),
        "weekly_sales": (
            open_df.groupby("Date")["Sales"].sum()
            .resample("W").sum()
            .reset_index()
            .rename(columns={"Date": "Week", "Sales": "Total Sales"})
        ),
    }


def models_trained() -> list[str]:
    found = []
    for arch in ["rnn", "lstm", "gru"]:
        if (MODELS_DIR / f"{arch}_best.pt").exists():
            found.append(arch.upper())
    return found


stats = get_summary_stats()
trained = models_trained()

st.title("Rossmann demand forecasting")
st.caption(
    "Time-series sales forecasting on 1M+ Rossmann retail records "
    "using RNN, LSTM, and GRU architectures · "
    f"{stats['date_min']} → {stats['date_max']}"
)

# ── KPI row ──────────────────────────────────────────────────────────────────
with st.container(horizontal=True):
    st.metric(
        "Total records",
        f"{stats['total_records']:,}",
        border=True,
    )
    st.metric(
        "Stores",
        f"{stats['n_stores']:,}",
        border=True,
    )
    st.metric(
        "Avg daily sales",
        f"€{stats['avg_sales']:,.0f}",
        border=True,
    )
    st.metric(
        "Promo days",
        f"{stats['promo_pct']:.1f}%",
        border=True,
    )
    st.metric(
        "Avg customers/day",
        f"{stats['avg_customers']:,.0f}",
        border=True,
    )

# ── Models trained status ──────────────────────────────────────────────────
st.subheader("Model training status")
if trained:
    cols = st.columns(3)
    arch_info = {
        "RNN": ("Vanilla RNN", "Baseline recurrent model", "gru_best.pt"),
        "LSTM": ("LSTM", "Long Short-Term Memory with layer norm", None),
        "GRU": ("GRU", "Gated Recurrent Unit — best MAPE ~10.2%", None),
    }
    for i, arch in enumerate(["RNN", "LSTM", "GRU"]):
        with cols[i]:
            is_trained = arch in trained
            with st.container(border=True):
                icon = ":material/check_circle:" if is_trained else ":material/radio_button_unchecked:"
                color = "green" if is_trained else "gray"
                st.markdown(f":{color}[{icon} **{arch}**]")
                st.caption(arch_info[arch][1])
                if is_trained:
                    st.badge("Trained", color="green", icon=":material/done:")
                else:
                    st.badge("Not trained", color="orange", icon=":material/pending:")
    if len(trained) < 3:
        st.info(
            "Some models are not trained yet. Go to **Train models** in the sidebar.",
            icon=":material/info:",
        )
else:
    st.warning(
        "No trained models found. Head to **Train models** in the sidebar to get started.",
        icon=":material/warning:",
    )

# ── Weekly sales trend ────────────────────────────────────────────────────
st.subheader("Weekly aggregate sales trend")
weekly = stats["weekly_sales"]
st.area_chart(weekly, x="Week", y="Total Sales", color="#60A5FA")

# ── Store type breakdown ──────────────────────────────────────────────────
st.subheader("Dataset at a glance")
col1, col2 = st.columns(2)
with col1:
    with st.container(border=True):
        st.markdown("**Store type distribution**")
        type_df = pd.DataFrame(
            list(stats["store_types"].items()), columns=["Store type", "Count"]
        ).sort_values("Count", ascending=False)
        st.dataframe(type_df, hide_index=True)

with col2:
    with st.container(border=True):
        st.markdown("**Project tech stack**")
        st.markdown("""
| Component | Technology |
|-----------|-----------|
| Data processing | Python · Pandas · SQL |
| Feature engineering | Rolling lags · Cyclical encoding |
| Models | RNN · LSTM · **GRU** (best) |
| Framework | PyTorch + MPS acceleration |
| Dashboard | Streamlit |
| Metric | MAPE (≈10.2% with GRU) |
""")
