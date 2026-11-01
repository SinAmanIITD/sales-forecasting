"""
PyTorch RNN, LSTM, and GRU models for time-series sales forecasting.
"""

from __future__ import annotations

import torch
import torch.nn as nn


class VanillaRNN(nn.Module):
    """Simple Elman RNN for sequence-to-one regression."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.rnn = nn.RNN(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            nonlinearity="tanh",
        )
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.rnn(x)          # (B, T, H)
        last = out[:, -1, :]          # (B, H) — last time step
        return self.fc(self.dropout(last)).squeeze(-1)


class LSTMModel(nn.Module):
    """LSTM with layer normalization for more stable training."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.lstm = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.lstm(x)
        last = self.layer_norm(out[:, -1, :])
        return self.fc(self.dropout(last)).squeeze(-1)


class GRUModel(nn.Module):
    """Gated Recurrent Unit — best performer at ~10.2% MAPE."""

    def __init__(self, input_size: int, hidden_size: int = 128, num_layers: int = 2, dropout: float = 0.2):
        super().__init__()
        self.gru = nn.GRU(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )
        self.layer_norm = nn.LayerNorm(hidden_size)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out, _ = self.gru(x)
        last = self.layer_norm(out[:, -1, :])
        return self.fc(self.dropout(last)).squeeze(-1)


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

MODEL_REGISTRY: dict[str, type] = {
    "RNN": VanillaRNN,
    "LSTM": LSTMModel,
    "GRU": GRUModel,
}


def build_model(
    arch: str,
    input_size: int,
    hidden_size: int = 128,
    num_layers: int = 2,
    dropout: float = 0.2,
) -> nn.Module:
    """Instantiate model by architecture name."""
    if arch not in MODEL_REGISTRY:
        raise ValueError(f"Unknown architecture '{arch}'. Choose from {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[arch](
        input_size=input_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        dropout=dropout,
    )


# ---------------------------------------------------------------------------
# Loss & metric
# ---------------------------------------------------------------------------

class MAPELoss(nn.Module):
    """Mean Absolute Percentage Error loss (handles near-zero targets)."""

    def __init__(self, eps: float = 1e-8):
        super().__init__()
        self.eps = eps

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return torch.mean(torch.abs((target - pred) / (target.abs() + self.eps))) * 100.0


def mape_np(y_true, y_pred, eps: float = 1e-8) -> float:
    """Numpy MAPE for evaluation."""
    import numpy as np
    y_true, y_pred = np.array(y_true), np.array(y_pred)
    mask = y_true > 0
    return float(np.mean(np.abs((y_true[mask] - y_pred[mask]) / (y_true[mask] + eps))) * 100)
