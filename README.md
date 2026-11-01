# Rossmann Time-Series Demand Forecasting

**Time-Series Demand Forecasting | Python, SQL, GRU/RNNs, Streamlit**

> Processed and cleaned 1M+ Rossmann retail records · Evaluated RNN, LSTM, and GRU architectures · Achieved **10.2% MAPE** with GRU · Interactive Streamlit dashboard

---

## Project Structure

```
SalesForcasting/
├── rossmann-store-sales/          # Raw dataset (1M+ records)
│   ├── train.csv
│   ├── store.csv
│   └── test.csv
│
├── src/
│   ├── data_pipeline.py           # Data loading, cleaning, feature engineering
│   ├── models.py                  # RNN, LSTM, GRU PyTorch architectures
│   ├── train.py                   # Training script with early stopping
│   ├── inference.py               # Model loading & autoregressive forecasting
│   └── sql_analysis.py            # Loads cleaned data into SQLite, runs sql/rossmann_analysis.sql
│
├── sql/
│   └── rossmann_analysis.sql      # Weekly trend, promo/holiday impact, top stores, competition-distance queries
│
├── app_pages/
│   ├── home.py                    # Overview dashboard with KPIs
│   ├── eda.py                     # Interactive data explorer
│   ├── forecast.py                # Store-level demand forecast
│   ├── model_eval.py              # Architecture comparison & training curves
│   └── train_runner.py            # Launch training from the UI
│
├── models/                        # Saved model checkpoints (auto-created)
│   ├── gru_best.pt
│   ├── lstm_best.pt
│   ├── rnn_best.pt
│   ├── scalers.pkl
│   └── training_summary.json
│
├── .streamlit/
│   └── config.toml                # Dark financial dashboard theme
│
├── streamlit_app.py               # Main Streamlit entry point
└── requirements.txt
```

---

## Quick Start

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Train models
```bash
# Train GRU only (fastest, best MAPE)
python3 src/train.py --arch GRU --epochs 20 --stores 100

# Train all three architectures
python3 src/train.py --arch all --epochs 20 --stores 100
```

### 3. Launch dashboard
```bash
streamlit run streamlit_app.py
```

---

## Model Architecture

| Model | Params | Best Validation MAPE | Notes |
|-------|--------|-----------------|-------|
| Vanilla RNN | 52,737 | 15.82% | Baseline; struggles with long dependencies, overfits past epoch 7 |
| LSTM | 210,817 | 12.15% | Layer norm stabilizes training |
| **GRU** | **158,209** | **10.21%** | **Best — fewer params than LSTM, better MAPE; optimal gating for promo spikes** |

(Param counts computed directly from the trained checkpoints, input_size=23. Note GRU has *fewer* parameters than LSTM — GRU's 2 gates vs. LSTM's 3 means less to learn per layer, which helps here given the dataset's size per store.)

All models use:
- 2 stacked layers, hidden size 128
- AdamW + ReduceLROnPlateau scheduler
- Huber loss (robust to outliers)
- Early stopping (patience = 5)
- Apple MPS acceleration

---

## Feature Engineering

| Feature | Description |
|---------|-------------|
| Rolling lags | Sales lag 1, 7, 14, 28 days |
| Rolling stats | 7/14/30-day mean & std |
| Cyclical encoding | Month sin/cos, DayOfWeek sin/cos |
| Store metadata | StoreType, Assortment dummies |
| Promotion | Promo flag, Promo × Customers interaction |
| Competition | CompetitionDistance (median-imputed where missing) |

---

## Dashboard Pages

| Page | Description |
|------|-------------|
| **Overview** | KPI cards, weekly trend, model status |
| **Data explorer** | Filter by store type, promo, date range; trend & distribution charts |
| **Store forecast** | 7–28 day ahead predictions, selectable store & model |
| **Model evaluation** | Training curves, MAPE comparison, epoch tables |
| **Train models** | Configure & launch training, command generator |

---

## SQL Analysis

```bash
python3 src/sql_analysis.py   # loads cleaned data into SQLite, runs sql/rossmann_analysis.sql, saves results to reports/
```

See [`sql/rossmann_analysis.sql`](sql/rossmann_analysis.sql) for the queries (weekly trend, promo/holiday impact, day-of-week seasonality, top stores, competition-distance analysis) and [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) for the real findings and business interpretation.

---

## Interview Prep

Read [`INTERVIEW_PREP.md`](INTERVIEW_PREP.md) before any interview where this project comes up — full pipeline walkthrough with every design decision justified, real numbers verified against the trained checkpoints, and a question bank with model answers.
