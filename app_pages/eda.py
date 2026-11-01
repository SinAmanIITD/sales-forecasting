"""
Data Explorer (EDA) page.
"""

import sys
from pathlib import Path
import streamlit as st
import pandas as pd
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import load_raw_data, merge_and_clean


@st.cache_data(ttl="1h", show_spinner="Loading data …")
def load_clean_df() -> pd.DataFrame:
    train, store = load_raw_data()
    return merge_and_clean(train, store)


df = load_clean_df()

st.title("Data explorer")
st.caption(f"Exploring {len(df):,} records across {df['Store'].nunique()} stores · {df['Date'].min().date()} → {df['Date'].max().date()}")

# ── Sidebar filters ────────────────────────────────────────────────────────
with st.sidebar:
    st.header(":material/filter_list: Filters")

    store_type = st.multiselect(
        "Store type",
        options=sorted(df["StoreType"].dropna().unique()),
        default=sorted(df["StoreType"].dropna().unique()),
    )
    promo_filter = st.segmented_control(
        "Promotion", options=["All", "Promo only", "No promo"], default="All"
    )
    date_range = st.date_input(
        "Date range",
        value=(df["Date"].min().date(), df["Date"].max().date()),
        min_value=df["Date"].min().date(),
        max_value=df["Date"].max().date(),
    )

# Apply filters
filtered = df[df["StoreType"].isin(store_type)]
if promo_filter == "Promo only":
    filtered = filtered[filtered["Promo"] == 1]
elif promo_filter == "No promo":
    filtered = filtered[filtered["Promo"] == 0]

if len(date_range) == 2:
    start, end = pd.Timestamp(date_range[0]), pd.Timestamp(date_range[1])
    filtered = filtered[(filtered["Date"] >= start) & (filtered["Date"] <= end)]

# ── KPI row ────────────────────────────────────────────────────────────────
with st.container(horizontal=True):
    st.metric("Records", f"{len(filtered):,}", border=True)
    st.metric("Stores", f"{filtered['Store'].nunique():,}", border=True)
    st.metric("Avg sales", f"€{filtered['Sales'].mean():,.0f}", border=True)
    st.metric("Median sales", f"€{filtered['Sales'].median():,.0f}", border=True)

# ── Tabs ────────────────────────────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    ":material/timeline: Sales trends",
    ":material/bar_chart: Distribution",
    ":material/store: Store analysis",
    ":material/table_chart: Raw data",
])

with tab1:
    st.subheader("Daily aggregate sales")
    daily = (
        filtered.groupby("Date")[["Sales", "Customers"]]
        .sum()
        .reset_index()
        .rename(columns={"Sales": "Total sales", "Customers": "Total customers"})
    )
    weekly = (
        filtered.set_index("Date")
        .resample("W")["Sales"]
        .agg(["sum", "mean"])
        .reset_index()
        .rename(columns={"sum": "Weekly total", "mean": "Weekly avg"})
    )

    view = st.segmented_control("View", ["Daily", "Weekly"], default="Weekly")
    if view == "Daily":
        st.line_chart(daily, x="Date", y="Total sales")
    else:
        st.line_chart(weekly, x="Date", y=["Weekly total", "Weekly avg"])

    st.subheader("Sales by day of week")
    dow_map = {1: "Mon", 2: "Tue", 3: "Wed", 4: "Thu", 5: "Fri", 6: "Sat", 7: "Sun"}
    dow_df = (
        filtered.groupby("DayOfWeek")["Sales"].mean()
        .reset_index()
        .rename(columns={"DayOfWeek": "Day", "Sales": "Avg sales"})
    )
    dow_df["Day"] = dow_df["Day"].map(dow_map)
    st.bar_chart(dow_df, x="Day", y="Avg sales", color="#60A5FA")

with tab2:
    st.subheader("Sales distribution")
    col1, col2 = st.columns(2)
    with col1:
        with st.container(border=True):
            st.markdown("**By promotion status**")
            promo_df = (
                filtered.groupby("Promo")["Sales"].mean()
                .reset_index()
                .rename(columns={"Promo": "Promotion", "Sales": "Avg sales"})
            )
            promo_df["Promotion"] = promo_df["Promotion"].map({0: "No promo", 1: "Promo active"})
            st.bar_chart(promo_df, x="Promotion", y="Avg sales", color="#34D399")

    with col2:
        with st.container(border=True):
            st.markdown("**By holiday type**")
            holiday_map = {"0": "None", "a": "Public holiday", "b": "Easter", "c": "Christmas"}
            hol_df = (
                filtered.groupby("StateHoliday")["Sales"].mean()
                .reset_index()
                .rename(columns={"StateHoliday": "Holiday", "Sales": "Avg sales"})
            )
            hol_df["Holiday"] = hol_df["Holiday"].map(
                lambda x: "None" if str(x) in ("0", "False") else str(x)
            )
            st.bar_chart(hol_df, x="Holiday", y="Avg sales", color="#A78BFA")

    st.subheader("Monthly seasonality")
    month_df = (
        filtered.groupby(filtered["Date"].dt.month)["Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Date": "Month", "Sales": "Avg sales"})
    )
    month_names = {1: "Jan", 2: "Feb", 3: "Mar", 4: "Apr", 5: "May", 6: "Jun",
                   7: "Jul", 8: "Aug", 9: "Sep", 10: "Oct", 11: "Nov", 12: "Dec"}
    month_df["Month"] = month_df["Month"].map(month_names)
    st.bar_chart(month_df, x="Month", y="Avg sales", color="#FBBF24")

with tab3:
    st.subheader("Top & bottom performing stores")
    store_perf = (
        filtered.groupby("Store")["Sales"]
        .agg(["mean", "sum", "std"])
        .reset_index()
        .rename(columns={"mean": "Avg sales", "sum": "Total sales", "std": "Std dev"})
        .sort_values("Avg sales", ascending=False)
    )

    sort_by = st.segmented_control("Sort by", ["Top 20", "Bottom 20"], default="Top 20")
    show_df = store_perf.head(20) if sort_by == "Top 20" else store_perf.tail(20)
    st.bar_chart(show_df, x="Store", y="Avg sales", color="#60A5FA")

    st.subheader("Store type comparison")
    type_df = (
        filtered.groupby("StoreType")["Sales"]
        .mean()
        .reset_index()
        .rename(columns={"Sales": "Avg sales"})
    )
    st.bar_chart(type_df, x="StoreType", y="Avg sales", color="#34D399")

with tab4:
    st.subheader("Raw data sample")
    n_rows = st.slider("Rows to show", 100, 5000, 500, step=100)
    st.dataframe(
        filtered.sort_values("Date", ascending=False)
        .head(n_rows)
        .reset_index(drop=True),
        hide_index=True,
    )
