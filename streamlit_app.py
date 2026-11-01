"""
Rossmann Time-Series Demand Forecasting Dashboard
Main entry point — multi-page app via st.navigation.
"""

import streamlit as st

st.set_page_config(
    page_title="Rossmann Demand Forecasting",
    page_icon=":material/monitoring:",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Pages ─────────────────────────────────────────────────────────────────
home_page = st.Page("app_pages/home.py", title="Overview", icon=":material/home:", default=True)
eda_page = st.Page("app_pages/eda.py", title="Data explorer", icon=":material/bar_chart:")
forecast_page = st.Page("app_pages/forecast.py", title="Store forecast", icon=":material/trending_up:")
model_page = st.Page("app_pages/model_eval.py", title="Model evaluation", icon=":material/model_training:")
train_page = st.Page("app_pages/train_runner.py", title="Train models", icon=":material/play_circle:")

pg = st.navigation([home_page, eda_page, forecast_page, model_page, train_page])
pg.run()
