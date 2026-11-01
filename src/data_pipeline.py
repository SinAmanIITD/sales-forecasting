"""
Data pipeline for Rossmann Store Sales demand forecasting.
Handles loading, cleaning, feature engineering, and sequence preparation.
"""

from __future__ import annotations

import os
import pickle
from pathlib import Path
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


DATA_DIR = Path(__file__).resolve().parent.parent / "rossmann-store-sales"
MODELS_DIR = Path(__file__).resolve().parent.parent / "models"
SEQ_LEN = 30  # look-back window (days)
FORECAST_HORIZON = 7  # days ahead to predict


# ---------------------------------------------------------------------------
# 1. Load & merge raw data
# ---------------------------------------------------------------------------

def load_raw_data() -> tuple[pd.DataFrame, pd.DataFrame]:
    """Load train.csv and store.csv from disk."""
    train = pd.read_csv(DATA_DIR / "train.csv", low_memory=False, parse_dates=["Date"])
    store = pd.read_csv(DATA_DIR / "store.csv")
    return train, store


def merge_and_clean(train: pd.DataFrame, store: pd.DataFrame) -> pd.DataFrame:
    """Merge store metadata, clean, and filter for open stores with positive sales."""
    df = train.merge(store, on="Store", how="left")

    # Keep only open stores with sales > 0
    df = df[(df["Open"] == 1) & (df["Sales"] > 0)].copy()

    # Parse StateHoliday: '0' → 0, else 1
    df["StateHoliday"] = df["StateHoliday"].apply(lambda x: 0 if str(x) == "0" else 1)

    # Fill missing competition distance with large value (no competitor)
    df["CompetitionDistance"] = df["CompetitionDistance"].fillna(df["CompetitionDistance"].median())

    # Fill Promo2 related nulls
    for col in ["Promo2SinceWeek", "Promo2SinceYear"]:
        df[col] = df[col].fillna(0)

    # Fill competition open dates and promo interval
    df["CompetitionOpenSinceMonth"] = df["CompetitionOpenSinceMonth"].fillna(0)
    df["CompetitionOpenSinceYear"] = df["CompetitionOpenSinceYear"].fillna(0)
    df["PromoInterval"] = df["PromoInterval"].fillna("None")

    df = df.sort_values(["Store", "Date"]).reset_index(drop=True)
    return df


# ---------------------------------------------------------------------------
# 2. Feature engineering
# ---------------------------------------------------------------------------

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add time-based, lag, and rolling features."""
    df = df.copy()

    # --- Date features ---
    df["Year"] = df["Date"].dt.year
    df["Month"] = df["Date"].dt.month
    df["Week"] = df["Date"].dt.isocalendar().week.astype(int)
    df["DayOfYear"] = df["Date"].dt.dayofyear

    # Cyclical encoding for seasonality
    df["Month_sin"] = np.sin(2 * np.pi * df["Month"] / 12)
    df["Month_cos"] = np.cos(2 * np.pi * df["Month"] / 12)
    df["DayOfWeek_sin"] = np.sin(2 * np.pi * df["DayOfWeek"] / 7)
    df["DayOfWeek_cos"] = np.cos(2 * np.pi * df["DayOfWeek"] / 7)

    # --- Store-level lag features (within each store) ---
    df = df.sort_values(["Store", "Date"])
    for lag in [1, 7, 14, 28]:
        df[f"Sales_lag_{lag}"] = df.groupby("Store")["Sales"].shift(lag)

    # --- Rolling window features ---
    for window in [7, 14, 30]:
        df[f"Sales_roll_mean_{window}"] = (
            df.groupby("Store")["Sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).mean())
        )
        df[f"Sales_roll_std_{window}"] = (
            df.groupby("Store")["Sales"]
            .transform(lambda x: x.shift(1).rolling(window, min_periods=1).std().fillna(0))
        )

    # --- Promo interaction ---
    df["Promo_x_Customers"] = df["Promo"] * df["Customers"]

    # --- Store type dummies ---
    df = pd.get_dummies(df, columns=["StoreType", "Assortment"], drop_first=True, dtype=float)

    # Drop rows with NaN from lag/rolling creation only (preserves all stores)
    lag_cols = [c for c in df.columns if "Sales_lag_" in c or "Sales_roll_" in c]
    df = df.dropna(subset=lag_cols).reset_index(drop=True)

    return df


# ---------------------------------------------------------------------------
# 3. Sequence builder for RNN/LSTM/GRU
# ---------------------------------------------------------------------------

FEATURE_COLS = [
    "Month_sin", "Month_cos", "DayOfWeek_sin", "DayOfWeek_cos",
    "Year", "Week", "DayOfYear",
    "Promo", "StateHoliday", "SchoolHoliday",
    "Promo2", "CompetitionDistance",
    "Sales_lag_1", "Sales_lag_7", "Sales_lag_14", "Sales_lag_28",
    "Sales_roll_mean_7", "Sales_roll_mean_14", "Sales_roll_mean_30",
    "Sales_roll_std_7", "Sales_roll_std_14", "Sales_roll_std_30",
    "Promo_x_Customers",
]

TARGET_COL = "Sales"


def build_sequences(
    store_df: pd.DataFrame,
    scaler_X: MinMaxScaler | None = None,
    scaler_y: MinMaxScaler | None = None,
    fit_scalers: bool = False,
    seq_len: int = SEQ_LEN,
) -> tuple[np.ndarray, np.ndarray, MinMaxScaler, MinMaxScaler]:
    """Build (X, y) sequences for a single store's time series."""
    # Keep only columns that exist in this frame
    feat_cols = [c for c in FEATURE_COLS if c in store_df.columns]

    X_raw = store_df[feat_cols].values.astype(np.float32)
    y_raw = store_df[[TARGET_COL]].values.astype(np.float32)

    if fit_scalers or scaler_X is None:
        scaler_X = MinMaxScaler()
        X_scaled = scaler_X.fit_transform(X_raw)
    else:
        X_scaled = scaler_X.transform(X_raw)

    if fit_scalers or scaler_y is None:
        scaler_y = MinMaxScaler()
        y_scaled = scaler_y.fit_transform(y_raw)
    else:
        y_scaled = scaler_y.transform(y_raw)

    X_seqs, y_seqs = [], []
    for i in range(seq_len, len(X_scaled)):
        X_seqs.append(X_scaled[i - seq_len : i])
        y_seqs.append(y_scaled[i, 0])

    return np.array(X_seqs), np.array(y_seqs), scaler_X, scaler_y


# ---------------------------------------------------------------------------
# 4. Train/val split and multi-store data builder
# ---------------------------------------------------------------------------

def get_processed_df() -> pd.DataFrame:
    """Full pipeline: load → clean → engineer features."""
    train, store = load_raw_data()
    df = merge_and_clean(train, store)
    df = engineer_features(df)
    return df


def build_multi_store_dataset(
    df: pd.DataFrame,
    store_ids: list[int],
    val_ratio: float = 0.15,
    seq_len: int = SEQ_LEN,
) -> dict:
    """Build train/val tensors for a list of stores."""
    import torch

    all_X_train, all_y_train = [], []
    all_X_val, all_y_val = [], []
    scalers: dict[int, dict] = {}

    for sid in store_ids:
        sdf = df[df["Store"] == sid].sort_values("Date").reset_index(drop=True)
        if len(sdf) < seq_len + 10:
            continue

        split = int(len(sdf) * (1 - val_ratio))
        train_df = sdf.iloc[:split]
        val_df = sdf.iloc[split - seq_len:]  # overlap by seq_len for continuity

        X_tr, y_tr, sx, sy = build_sequences(train_df, fit_scalers=True, seq_len=seq_len)
        X_vl, y_vl, _, _ = build_sequences(val_df, scaler_X=sx, scaler_y=sy, seq_len=seq_len)

        scalers[sid] = {"scaler_X": sx, "scaler_y": sy}
        all_X_train.append(X_tr)
        all_y_train.append(y_tr)
        all_X_val.append(X_vl)
        all_y_val.append(y_vl)

    X_train = torch.tensor(np.concatenate(all_X_train), dtype=torch.float32)
    y_train = torch.tensor(np.concatenate(all_y_train), dtype=torch.float32)
    X_val = torch.tensor(np.concatenate(all_X_val), dtype=torch.float32)
    y_val = torch.tensor(np.concatenate(all_y_val), dtype=torch.float32)

    n_features = X_train.shape[2]

    return {
        "X_train": X_train,
        "y_train": y_train,
        "X_val": X_val,
        "y_val": y_val,
        "scalers": scalers,
        "n_features": n_features,
    }


def save_scalers(scalers: dict, path: Path | None = None) -> None:
    path = path or (MODELS_DIR / "scalers.pkl")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(scalers, f)


def load_scalers(path: Path | None = None) -> dict:
    path = path or (MODELS_DIR / "scalers.pkl")
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# 5. Inference helper
# ---------------------------------------------------------------------------

def get_store_recent_sequence(
    df: pd.DataFrame,
    store_id: int,
    scaler_X: MinMaxScaler,
    seq_len: int = SEQ_LEN,
) -> np.ndarray:
    """Return the most recent `seq_len` rows for a store, scaled."""
    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    sdf = df[df["Store"] == store_id].sort_values("Date").tail(seq_len)
    X = sdf[feat_cols].values.astype(np.float32)
    return scaler_X.transform(X)
