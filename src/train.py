"""
Training script for Rossmann demand forecasting.
Trains RNN, LSTM, and GRU models, saves weights + metrics.

Usage:
    python src/train.py                      # train all models
    python src/train.py --arch GRU           # train only GRU
    python src/train.py --arch GRU --epochs 30 --stores 50
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

# ── allow running as a script from any location ──────────────────────────────
sys.path.insert(0, str(Path(__file__).resolve().parent))

from data_pipeline import (
    MODELS_DIR,
    get_processed_df,
    build_multi_store_dataset,
    save_scalers,
)
from models import build_model, mape_np

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------
DEFAULT_CONFIG = {
    "hidden_size": 128,
    "num_layers": 2,
    "dropout": 0.2,
    "epochs": 20,
    "batch_size": 512,
    "lr": 1e-3,
    "weight_decay": 1e-5,
    "patience": 5,          # early-stopping patience
    "val_ratio": 0.15,
    "num_stores": 1115,     # 1,115 stores covers the full Rossmann store network
    "seq_len": 30,
}


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    criterion: nn.Module,
    device: torch.device,
) -> float:
    model.train()
    total_loss = 0.0
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        optimizer.zero_grad()
        pred = model(X_batch)
        loss = criterion(pred, y_batch)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item() * len(y_batch)
    return total_loss / len(loader.dataset)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device,
    scaler_y_sample,
) -> tuple[float, float]:
    """Return (val_loss, val_MAPE_on_original_scale)."""
    model.eval()
    total_loss = 0.0
    all_pred, all_true = [], []
    for X_batch, y_batch in loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        pred = model(X_batch)
        total_loss += criterion(pred, y_batch).item() * len(y_batch)
        all_pred.append(pred.cpu().numpy())
        all_true.append(y_batch.cpu().numpy())

    all_pred = np.concatenate(all_pred).reshape(-1, 1)
    all_true = np.concatenate(all_true).reshape(-1, 1)

    # Inverse-transform using a sample scaler (all stores were fit similarly)
    pred_inv = scaler_y_sample.inverse_transform(all_pred).ravel()
    true_inv = scaler_y_sample.inverse_transform(all_true).ravel()
    mape = mape_np(true_inv, pred_inv)

    return total_loss / len(loader.dataset), mape


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def train_model(
    arch: str,
    dataset: dict,
    config: dict,
    device: torch.device,
    save_dir: Path,
) -> dict:
    print(f"\n{'='*60}")
    print(f"  Training {arch}")
    print(f"{'='*60}")

    X_train, y_train = dataset["X_train"], dataset["y_train"]
    X_val, y_val = dataset["X_val"], dataset["y_val"]
    n_features = dataset["n_features"]

    train_ds = TensorDataset(X_train, y_train)
    val_ds = TensorDataset(X_val, y_val)
    train_loader = DataLoader(train_ds, batch_size=config["batch_size"], shuffle=True, num_workers=0)
    val_loader = DataLoader(val_ds, batch_size=config["batch_size"] * 2, shuffle=False, num_workers=0)

    model = build_model(
        arch,
        input_size=n_features,
        hidden_size=config["hidden_size"],
        num_layers=config["num_layers"],
        dropout=config["dropout"],
    ).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(), lr=config["lr"], weight_decay=config["weight_decay"]
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=3
    )
    criterion = nn.HuberLoss()  # robust to outliers vs MSE

    # Sample scaler for inverse-transform during eval
    sample_scalers = dataset["scalers"]
    first_sid = next(iter(sample_scalers))
    sample_scaler_y = sample_scalers[first_sid]["scaler_y"]

    best_val_mape = float("inf")
    best_state = None
    history: list[dict] = []
    patience_count = 0
    t0 = time.time()

    for epoch in range(1, config["epochs"] + 1):
        train_loss = train_one_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_mape = evaluate(model, val_loader, criterion, device, sample_scaler_y)
        scheduler.step(val_loss)

        lr_now = optimizer.param_groups[0]["lr"]
        history.append({"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "val_mape": val_mape})
        print(f"  Epoch {epoch:>3}/{config['epochs']} | train_loss={train_loss:.5f} | val_loss={val_loss:.5f} | MAPE={val_mape:.2f}% | lr={lr_now:.6f}")

        if val_mape < best_val_mape:
            best_val_mape = val_mape
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            patience_count = 0
        else:
            patience_count += 1
            if patience_count >= config["patience"]:
                print(f"  Early stop at epoch {epoch}.")
                break

    elapsed = time.time() - t0
    print(f"\n  Best {arch} val MAPE = {best_val_mape:.2f}%  ({elapsed:.1f}s)")

    # Save best weights
    save_dir.mkdir(parents=True, exist_ok=True)
    model.load_state_dict(best_state)
    torch.save(model.state_dict(), save_dir / f"{arch.lower()}_best.pt")

    return {"arch": arch, "best_val_mape": best_val_mape, "history": history}


def main():
    parser = argparse.ArgumentParser(description="Train RNN/LSTM/GRU for Rossmann forecasting")
    parser.add_argument("--arch", default="all", choices=["RNN", "LSTM", "GRU", "all"])
    parser.add_argument("--epochs", type=int, default=DEFAULT_CONFIG["epochs"])
    parser.add_argument("--stores", type=int, default=DEFAULT_CONFIG["num_stores"])
    parser.add_argument("--batch_size", type=int, default=DEFAULT_CONFIG["batch_size"])
    parser.add_argument("--lr", type=float, default=DEFAULT_CONFIG["lr"])
    parser.add_argument("--hidden", type=int, default=DEFAULT_CONFIG["hidden_size"])
    args = parser.parse_args()

    config = DEFAULT_CONFIG.copy()
    config["epochs"] = args.epochs
    config["num_stores"] = args.stores
    config["batch_size"] = args.batch_size
    config["lr"] = args.lr
    config["hidden_size"] = args.hidden

    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    print(f"Device: {device}")

    print("\nLoading and processing data …")
    df = get_processed_df()

    # Select stores
    all_stores = sorted(df["Store"].unique())
    if config["num_stores"] and config["num_stores"] < len(all_stores):
        # Pick stores with most data points for training stability
        store_counts = df.groupby("Store").size().sort_values(ascending=False)
        store_ids = store_counts.index[: config["num_stores"]].tolist()
    else:
        store_ids = all_stores

    print(f"Using {len(store_ids)} stores for training …")
    dataset = build_multi_store_dataset(
        df, store_ids, val_ratio=config["val_ratio"], seq_len=config["seq_len"]
    )
    print(f"  X_train: {dataset['X_train'].shape}, X_val: {dataset['X_val'].shape}")
    print(f"  Features: {dataset['n_features']}")

    save_scalers(dataset["scalers"])

    archs = ["RNN", "LSTM", "GRU"] if args.arch == "all" else [args.arch]
    all_results = []
    for arch in archs:
        result = train_model(arch, dataset, config, device, MODELS_DIR)
        all_results.append(result)

    # Save summary
    summary = {
        "config": config,
        "store_ids": store_ids,
        "n_features": dataset["n_features"],
        "results": [
            {"arch": r["arch"], "best_val_mape": r["best_val_mape"],
             "history": r["history"]}
            for r in all_results
        ],
    }
    with open(MODELS_DIR / "training_summary.json", "w") as f:
        json.dump(summary, f, indent=2)

    print("\n\n📊 Final Results:")
    print("-" * 40)
    for r in all_results:
        print(f"  {r['arch']:6s} → MAPE = {r['best_val_mape']:.2f}%")
    print(f"\nModels saved to: {MODELS_DIR}")


if __name__ == "__main__":
    main()
