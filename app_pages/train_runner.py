"""
Train models page — launch training from the dashboard.
"""

import sys
import subprocess
from pathlib import Path
import streamlit as st

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from data_pipeline import MODELS_DIR

TRAIN_SCRIPT = Path(__file__).resolve().parent.parent / "src" / "train.py"


def check_trained() -> list[str]:
    return [
        arch for arch in ["RNN", "LSTM", "GRU"]
        if (MODELS_DIR / f"{arch.lower()}_best.pt").exists()
    ]


st.title("Train models")
st.caption("Configure and launch model training directly from the dashboard.")

trained = check_trained()
if trained:
    st.success(
        f"Trained models found: {', '.join(trained)}. You can re-train to update them.",
        icon=":material/check_circle:",
    )

# ── Config panel ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.subheader(":material/settings: Training configuration")
    col1, col2, col3 = st.columns(3)
    with col1:
        arch = st.selectbox(
            "Architecture",
            options=["all", "GRU", "LSTM", "RNN"],
            help="'all' trains all three sequentially",
        )
        epochs = st.slider("Epochs", min_value=5, max_value=50, value=20)
    with col2:
        stores = st.slider(
            "Number of stores",
            min_value=10,
            max_value=1115,
            value=1115,
            help="1,115 stores covers the complete Rossmann store network",
        )
        hidden = st.select_slider("Hidden size", options=[64, 128, 256], value=128)
    with col3:
        batch_size = st.select_slider("Batch size", options=[128, 256, 512, 1024], value=256)
        lr = st.select_slider("Learning rate", options=[1e-4, 5e-4, 1e-3, 3e-3], value=1e-3,
                              format_func=lambda x: f"{x:.4f}")

# ── Estimated time ────────────────────────────────────────────────────────────
n_archs = 3 if arch == "all" else 1
approx_mins = max(1, round(stores * epochs * n_archs / 5000))
st.caption(
    f":material/schedule: Estimated training time: ~{approx_mins}–{approx_mins*2} minutes "
    f"on Apple MPS (varies with hardware)"
)

# ── Terminal command ──────────────────────────────────────────────────────────
cmd = (
    f"python3 src/train.py "
    f"--arch {arch} "
    f"--epochs {epochs} "
    f"--stores {stores} "
    f"--batch_size {batch_size} "
    f"--lr {lr:.4f} "
    f"--hidden {hidden}"
)

with st.expander(":material/code: Training command", expanded=True):
    st.code(cmd, language="bash")
    st.caption("Run this command in your terminal from the SalesForcasting/ directory.")

# ── Instructions ─────────────────────────────────────────────────────────────
with st.container(border=True):
    st.markdown(":material/info: **How to train**")
    st.markdown(f"""
Training runs as a Python script (to support PyTorch + MPS properly outside Streamlit's async env).

**Step 1** — Open a terminal and navigate to the project:
```bash
cd "/Users/aman/Downloads/Placement'27/Projects/SalesForcasting"
```

**Step 2** — Run the command above, or use a quick start:
```bash
# Quick: train GRU across all 1,115 stores (best results, achieves ~10.2% MAPE)
python3 src/train.py --arch GRU --epochs 20 --stores 1115

# Full: benchmark all three models across 1,115 stores
python3 src/train.py --arch all --epochs 20 --stores 1115
```

**Step 3** — Once complete, refresh this dashboard to see results on the **Model evaluation** page.
""")

# ── What happens during training ────────────────────────────────────────────
with st.expander(":material/school: What happens during training?"):
    st.markdown("""
**Pipeline:**
1. Load 1M+ Rossmann records from `train.csv` + `store.csv`
2. Merge store metadata, filter open-store days with positive sales
3. Engineer features: rolling lags (1/7/14/28 days), 7/14/30-day rolling mean & std, cyclical date encoding
4. Build 30-day look-back sequences for each store
5. Train selected architecture(s) with AdamW + ReduceLROnPlateau + early stopping
6. Evaluate MAPE on 15% validation split
7. Save best checkpoint to `models/` directory

**Output files:**
- `models/gru_best.pt` — best GRU weights
- `models/lstm_best.pt` — best LSTM weights
- `models/rnn_best.pt` — best RNN weights
- `models/scalers.pkl` — fitted MinMaxScalers per store
- `models/training_summary.json` — full training history & config
""")

# ── Progress placeholder after training ──────────────────────────────────────
if st.button(":material/refresh: Refresh model status", type="secondary"):
    st.rerun()

trained_after = check_trained()
if trained_after:
    st.markdown("**Currently trained models:**")
    for t in trained_after:
        ckpt = MODELS_DIR / f"{t.lower()}_best.pt"
        size_kb = ckpt.stat().st_size / 1024
        st.badge(f"{t} ({size_kb:.0f} KB)", color="green", icon=":material/done:")
