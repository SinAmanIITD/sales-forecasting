"""
Inference helpers: load trained models and generate forecasts.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data_pipeline import (
    MODELS_DIR,
    FEATURE_COLS,
    SEQ_LEN,
    get_processed_df,
    load_scalers,
    get_store_recent_sequence,
)
from models import build_model, mape_np


def load_trained_model(
    arch: str,
    n_features: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.0,
    device: torch.device | None = None,
) -> torch.nn.Module:
    """Load a saved model checkpoint."""
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = build_model(arch, n_features, hidden_size, num_layers, dropout)
    ckpt_path = MODELS_DIR / f"{arch.lower()}_best.pt"
    if not ckpt_path.exists():
        raise FileNotFoundError(f"No checkpoint found at {ckpt_path}. Run train.py first.")
    model.load_state_dict(torch.load(ckpt_path, map_location=device))
    model.to(device)
    model.eval()
    return model


def load_training_summary() -> dict | None:
    path = MODELS_DIR / "training_summary.json"
    if not path.exists():
        return None
    with open(path) as f:
        return json.load(f)


def get_store_scalers(df: pd.DataFrame, store_id: int, saved_scalers: dict | None = None):
    """
    Get scalers for a store. If store was in training set, uses saved scalers.
    Otherwise, fits scalers directly on the store's historical data so features
    and inverse transforms are perfectly calibrated to that store's distribution.
    """
    if saved_scalers and store_id in saved_scalers:
        return saved_scalers[store_id]["scaler_X"], saved_scalers[store_id]["scaler_y"]

    from sklearn.preprocessing import MinMaxScaler
    sdf = df[df["Store"] == store_id].sort_values("Date").reset_index(drop=True)
    feat_cols = [c for c in FEATURE_COLS if c in sdf.columns]

    scaler_X = MinMaxScaler()
    scaler_y = MinMaxScaler()
    scaler_X.fit(sdf[feat_cols].values.astype(np.float32))
    scaler_y.fit(sdf[["Sales"]].values.astype(np.float32))
    return scaler_X, scaler_y


def predict_store_next_n(
    model: torch.nn.Module,
    df: pd.DataFrame,
    store_id: int,
    scaler_X,
    scaler_y,
    n_steps: int = 7,
    seq_len: int = SEQ_LEN,
    device: torch.device | None = None,
) -> np.ndarray:
    """
    Autoregressively predict next `n_steps` sales for a store.
    Uses calendar progression, lag updating, and store historical context.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    store_df = df[df["Store"] == store_id].sort_values("Date").reset_index(drop=True)

    # Seed sequence from recent actual data
    seed_X = get_store_recent_sequence(df, store_id, scaler_X, seq_len)  # (seq_len, F)
    recent_X = list(seed_X)
    recent_sales = list(store_df["Sales"].tail(35).values)
    last_date = store_df["Date"].max()

    predictions = []
    with torch.no_grad():
        for step in range(n_steps):
            cur_date = last_date + pd.Timedelta(days=step + 1)
            x_window = np.array(recent_X[-seq_len:], dtype=np.float32)
            x_t = torch.tensor(x_window[np.newaxis], dtype=torch.float32).to(device)
            pred_scaled = model(x_t).item()
            pred_sales = float(scaler_y.inverse_transform([[pred_scaled]])[0][0])
            pred_sales = max(0.0, pred_sales)
            predictions.append(pred_sales)

            # Build next day's feature vector in raw space
            dow = cur_date.dayofweek + 1
            month = cur_date.month
            raw_row = {
                "Month_sin": np.sin(2 * np.pi * month / 12),
                "Month_cos": np.cos(2 * np.pi * month / 12),
                "DayOfWeek_sin": np.sin(2 * np.pi * dow / 7),
                "DayOfWeek_cos": np.cos(2 * np.pi * dow / 7),
                "Year": cur_date.year,
                "Week": cur_date.isocalendar()[1],
                "DayOfYear": cur_date.timetuple().tm_yday,
                "Promo": 1 if dow <= 5 else 0,
                "StateHoliday": 0,
                "SchoolHoliday": 0,
                "Promo2": store_df["Promo2"].iloc[-1] if "Promo2" in store_df.columns else 0,
                "CompetitionDistance": (
                    store_df["CompetitionDistance"].iloc[-1]
                    if "CompetitionDistance" in store_df.columns
                    else 1000.0
                ),
            }

            recent_sales.append(pred_sales)
            raw_row["Sales_lag_1"] = recent_sales[-2]
            raw_row["Sales_lag_7"] = recent_sales[-8] if len(recent_sales) >= 8 else recent_sales[-2]
            raw_row["Sales_lag_14"] = recent_sales[-15] if len(recent_sales) >= 15 else recent_sales[-2]
            raw_row["Sales_lag_28"] = recent_sales[-29] if len(recent_sales) >= 29 else recent_sales[-2]

            raw_row["Sales_roll_mean_7"] = np.mean(recent_sales[-8:-1])
            raw_row["Sales_roll_mean_14"] = np.mean(recent_sales[-15:-1])
            raw_row["Sales_roll_mean_30"] = np.mean(recent_sales[-31:-1])
            raw_row["Sales_roll_std_7"] = np.std(recent_sales[-8:-1])
            raw_row["Sales_roll_std_14"] = np.std(recent_sales[-15:-1])
            raw_row["Sales_roll_std_30"] = np.std(recent_sales[-31:-1])
            avg_cust = store_df["Customers"].mean() if "Customers" in store_df.columns else 600.0
            raw_row["Promo_x_Customers"] = raw_row["Promo"] * avg_cust

            row_vec = np.array([[raw_row.get(c, 0.0) for c in feat_cols]], dtype=np.float32)
            row_scaled = scaler_X.transform(row_vec)[0]
            recent_X.append(row_scaled)

    return np.array(predictions, dtype=np.float32)


def evaluate_model_on_test_window(
    model: torch.nn.Module,
    df: pd.DataFrame,
    store_ids: list[int],
    scaler_map: dict,
    test_days: int = 30,
    seq_len: int = SEQ_LEN,
    device: torch.device | None = None,
) -> pd.DataFrame:
    """
    Evaluate a model on the last `test_days` for each store.
    Returns a DataFrame with store-level MAPE.
    """
    if device is None:
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

    feat_cols = [c for c in FEATURE_COLS if c in df.columns]
    records = []
    model.eval()

    with torch.no_grad():
        for sid in store_ids:
            if sid not in scaler_map:
                continue
            sx = scaler_map[sid]["scaler_X"]
            sy = scaler_map[sid]["scaler_y"]
            sdf = df[df["Store"] == sid].sort_values("Date").reset_index(drop=True)

            if len(sdf) < seq_len + test_days:
                continue

            # Test window: last test_days rows
            test_start = len(sdf) - test_days
            preds_inv, trues_inv = [], []

            for i in range(test_start, len(sdf)):
                seed = sdf.iloc[i - seq_len : i][feat_cols].values.astype(np.float32)
                seed_sc = sx.transform(seed)
                x_t = torch.tensor(seed_sc[np.newaxis], dtype=torch.float32).to(device)
                pred_sc = model(x_t).item()
                pred_inv = sy.inverse_transform([[pred_sc]])[0][0]
                true_inv = sdf.iloc[i]["Sales"]
                preds_inv.append(pred_inv)
                trues_inv.append(true_inv)

            store_mape = mape_np(np.array(trues_inv), np.array(preds_inv))
            records.append({
                "Store": sid,
                "MAPE": store_mape,
                "Mean_Sales": np.mean(trues_inv),
                "Mean_Pred": np.mean(preds_inv),
            })

    return pd.DataFrame(records).sort_values("MAPE")
