# Interview Prep — Time-Series Demand Forecasting with RNNs
### IIT Delhi Placements 2026-27 — Analytics / Data Science Track

Read this cold, twice, before any interview where this project comes up. Project
code lives in this same folder; `app_pages/` is the Streamlit dashboard, `src/` is
the reusable pipeline, `sql/` is the SQL analysis layer, `models/` has the three
trained checkpoints and `training_summary.json` with the real per-epoch history
this document is built from.

---

## 1. The 30-Second Pitch

> "I built a demand forecasting system on the Rossmann dataset — just over a
> million daily store-sales records across 1,115 stores. I processed and
> explored it with both Python (pandas pipeline) and SQL (aggregate queries on
> a SQLite-loaded version of the cleaned data), engineered lag and rolling-window
> features plus cyclical date encodings, then compared three recurrent
> architectures — vanilla RNN, LSTM, and GRU — on 30-day lookback sequences. GRU
> won, hitting 10.2% validation MAPE, beating LSTM at 12.2% and plain RNN at
> 15.8%. I wrapped it all in an interactive Streamlit dashboard with store-level
> forecasts, an EDA explorer, and model comparison views."

---

## 2. Dataset

| Field | Value |
|---|---|
| Source | Rossmann Store Sales (Kaggle competition dataset) |
| Raw size | `train.csv` = 1,017,210 rows; `store.csv` = 1,115 stores (metadata) |
| After cleaning (open stores, Sales > 0) | 844,338 rows |
| Granularity | One row per (store, day) |
| Target | `Sales` (continuous, daily revenue per store) |

**Why "1M+ records" on the CV is accurate even though cleaning drops to
844K**: the CV bullet says "Processed 1M+ ... records," describing the raw input
volume the pipeline ingested and cleaned — not the final training set size. This
is a defensible, standard way to describe a pipeline's scale; be ready to explain
the 1.02M → 844K drop immediately if asked (see Section 3, Step 1).

---

## 3. The Pipeline, Step by Step — What and Why

### Step 1 — Cleaning (`merge_and_clean`, `src/data_pipeline.py`)
Merged `train.csv` with `store.csv` on `Store`, then filtered to **open stores
with `Sales > 0`** — dropping 1,017,210 → 844,338 rows (a 17% reduction).

**Why drop closed-store / zero-sales rows instead of keeping them?** A closed
store (`Open=0`) always has `Sales=0` by construction — it's not a demand
signal, it's a calendar fact (holiday closures, renovations). Including those
rows would teach the model "store closures predict zero sales," which is true
but useless — the business question is "how much will an *open* store sell,"
so training on closed-day rows would just dilute the sequence windows with
constant zeros and distort the MAPE metric (division by near-zero actuals).

Also handled: `StateHoliday` binarized (it's a categorical code with a
'0'-as-string footgun — `'0' != 0` in pandas, confirmed and fixed explicitly);
`CompetitionDistance` missing values filled with the median (not zero — a
missing distance means "unknown," not "zero distance / right next door," so
zero-filling would be a factual error); `Promo2SinceWeek/Year` and
`CompetitionOpenSinceMonth/Year` nulls filled with 0 (these genuinely mean "not
applicable," since the store never ran Promo2 — the null itself is
informative and 0 preserves that).

### Step 2 — SQL analysis layer (`sql/rossmann_analysis.sql`, `src/sql_analysis.py`)
This is the "**and SQL**" half of the CV's "Processed ... via Python and SQL."
The cleaned 844K-row frame is loaded into an in-memory SQLite table, and seven
aggregate queries run against it — weekly sales trend, promo impact by store
type, holiday impact, day-of-week seasonality, top-20 stores, a
competition-distance bucket analysis, and monthly seasonality. Real, run output
(not illustrative):

| Query | Key real finding |
|---|---|
| Promo impact by store type | Promo lifts average sales **26-43%** across every store type (e.g. type A: €5,809 → €8,305 with promo active) |
| Day-of-week seasonality | Monday (€8,216 avg) and Sunday (€8,225 avg) are the two highest-sales days; Saturday is lowest (€5,875) |
| Holiday impact | Days with both a state *and* school holiday average **€8,907** — *higher*, not lower, than a normal day (€6,896) — counterintuitive, see Section 6 |
| Competition distance | Stores with a competitor within 500m average **€7,615** — *higher* than stores 2-10km from any competitor (€6,697) |

**Why run SQL at all when pandas can do the same groupbys?** Two honest
reasons: (1) it's a second, independent verification path for the same
aggregates computed in the Streamlit EDA page — if pandas and SQL groupbys
agree, that's a real cross-check against a pandas indexing/grouping bug; (2) in
a real analytics role, exploratory aggregate queries like this usually run
directly against a warehouse table via SQL, not by pulling the full table into
a notebook first — this layer demonstrates that workflow on a dataset small
enough to also do in pandas.

### Step 3 — Feature engineering (`engineer_features`)
- **Cyclical encoding** (`sin`/`cos` of `Month` and `DayOfWeek`): a raw integer
  month (1-12) implies December (12) is "far" from January (1) in a model that
  treats it as a linear distance — cyclical encoding correctly places them
  adjacent on a circle, which matters for a *seasonal* business like retail.
- **Lag features** (`Sales_lag_{1,7,14,28}`): gives each day's row direct
  access to recent actual sales — 1-day lag captures short-term momentum,
  7/14/28-day lags capture weekly and bi-weekly/monthly cyclicality (Rossmann
  has strong day-of-week and pay-cycle effects).
- **Rolling mean/std** (7/14/30-day windows, **shifted by 1 day** before
  rolling — `x.shift(1).rolling(...)`): the shift is critical — without it, the
  rolling window would include the current day's own sales in its own feature,
  which is target leakage (the model would partially see the answer).
- **`Promo_x_Customers` interaction**: `Customers` alone doesn't explain a
  promo's full lift (foot traffic goes up *and* average basket size tends to
  rise during a promo) — the interaction term lets the model learn a
  promo-specific multiplier rather than assuming it applies uniformly.
- **One-hot `StoreType`/`Assortment`** (`drop_first=True`): low-cardinality
  nominal categoricals, no ordinal relationship — one-hot is correct, same
  reasoning as the churn project's contract/payment-method encoding.

Rows with NaN lag/rolling values (unavoidable at the start of each store's
time series) are dropped — this only removes the first 28 days *per store*,
not per the whole dataset, since lags are computed within each store's own
`groupby`.

### Step 4 — Sequence construction (`build_sequences`)
30-day lookback window (`SEQ_LEN=30`) → predicts the next single day's sales.
**Why 30 days, not longer?** It covers a full month (captures both weekly
cyclicality several times over and any monthly pay-cycle effect) while staying
short enough that 1,115 stores × ~750 days/store still yields enough training
sequences without excessive padding/truncation overhead; longer windows would
also increase vanishing-gradient risk for the plain RNN baseline, which is
useful to keep as a fair, strained baseline for comparison.

**Per-store scaling** (`MinMaxScaler`, fit separately per store): a flagship
store doing €20K/day and a small store doing €3K/day have very different
sales scales — a single global scaler would let the big stores dominate the
loss function. Per-store scaling puts every store's sequence on the same
[0,1] range before training, so the model learns *shape* (seasonality,
promo response) rather than *magnitude*, and magnitude is restored exactly by
inverse-transforming each store's own scaler at inference time.

### Step 5 — Train/val split (`build_multi_store_dataset`)
85/15 split, **chronological per store** (not random) — the validation set is
each store's most recent 15% of days, with a 30-day overlap into the training
tail so the first validation sequence still has a full lookback window. This
is the only correct way to split time-series data: a random row-level split
would let the model "see the future" (train on day 500, validate on day 300),
which silently inflates validation metrics and would make the reported MAPE
meaningless. **This is the single most important time-series-specific design
decision in this project — know it cold.**

Trained on **100 stores** (a representative subset of the 1,115 total, per
`training_summary.json`'s `store_ids` list) — chosen for training time
tractability on this hardware, not a dataset limitation; the pipeline supports
scaling to all 1,115 stores given more training time/compute (`--stores`
CLI flag in `src/train.py`).

### Step 6 — Model comparison: RNN vs. LSTM vs. GRU
All three share: 2 stacked layers, hidden size 128, `AdamW` optimizer,
`ReduceLROnPlateau` scheduler, a custom **MAPE loss** (not MSE — optimizing
MAPE directly, not as a reported-only metric, keeps training aligned with the
business metric a demand-planning team actually cares about), early stopping
(patience=5).

Real results, straight from `models/training_summary.json`:

| Model | Parameters | Best Val MAPE | Epoch of best | Behavior |
|---|---|---|---|---|
| Vanilla RNN | 52,737 | **15.82%** | 7 | Val MAPE stops improving after epoch 7, then climbs — classic overfitting/vanishing-gradient signature of a plain RNN on longer sequences |
| LSTM | 210,817 | **12.15%** | 8 | Smoother convergence, more parameters than GRU |
| **GRU** | **158,209** | **10.21%** | **10** | Best MAPE, *and* fewer parameters than LSTM |

**Why GRU won, specifically (not just "it did"):** GRU merges LSTM's forget
and input gates into a single update gate (2 gates vs. LSTM's 3), which means
less to learn per layer. On a dataset this size per store (roughly 700-950 days
per store, much smaller than, say, NLP-scale sequence data), LSTM's extra gate
capacity is more prone to overfitting before it's fully useful, while GRU's
leaner parameterization generalizes better — exactly the pattern seen here:
GRU beats LSTM on validation MAPE *while using ~25% fewer parameters*. This is
a genuinely instructive result, not just "the CV says GRU so I used GRU" — be
ready to make this exact argument if asked "why not just use LSTM, it's more
expressive?"

### Step 7 — The Streamlit dashboard
Five pages: **Overview** (KPI cards + weekly trend), **Data explorer** (the
pandas-side EDA — store type/promo/date filters, trend/distribution/store
tabs), **Store forecast** (7-28 day-ahead predictions via `src/inference.py`'s
autoregressive rollout — each predicted day feeds back in as the next day's
most recent lag), **Model evaluation** (training curves + MAPE comparison —
the source of the table in Step 6), **Train models** (launches training from
the UI, generates the equivalent CLI command).

**Why autoregressive multi-step forecasting, not a direct multi-output head?**
The model was trained as a single-step (next-day) predictor — extending that to
a 7-28 day forecast means feeding each prediction back in as a lag feature for
the next step. The honest tradeoff: errors compound over the horizon (day-28's
forecast is built on 27 prior *predictions*, not actuals), so forecast
uncertainty genuinely grows with horizon length — worth saying this
proactively if asked about the forecast page's accuracy at longer horizons,
rather than claiming uniform 10.2% MAPE out to day 28 (the reported 10.2% is
the *single-step*, next-day validation MAPE).

---

## 4. CV Bullet Points — Justified Line by Line

> **"Processed 1M+ Rossmann sales records via Python and SQL, preparing high-quality datasets for forecasting/analysis."**
- 1M+ → literal: 1,017,210 raw rows (`rossmann-store-sales/train.csv`).
- Python → `src/data_pipeline.py` (cleaning, feature engineering, sequence building).
- SQL → `sql/rossmann_analysis.sql` run via `src/sql_analysis.py` against a SQLite-loaded copy of the cleaned data (Section 3, Step 2) — seven real aggregate queries with genuine findings.

> **"Engineered RNN, LSTM, and GRU models; GRU achieved 10.2% MAPE in capturing weekly promotional/seasonal demand."**
- All three literally implemented in `src/models.py` and trained (`models/*.pt`).
- GRU's best validation MAPE = **10.21%** (`training_summary.json`, epoch 10) — rounds to 10.2%, exact match.
- "Weekly promotional/seasonal demand" → the day-of-week cyclical encoding + `Promo_x_Customers` interaction + lag-7/14/28 features are precisely what let the model capture weekly and promo-driven patterns (Section 3, Step 3).

> **"Developed a Streamlit dashboard integrating sales trends, store-level forecasts, and ML predictions for demand planning."**
- Fully built, running dashboard (`streamlit_app.py` + 5 pages in `app_pages/`) — sales trends (Overview + Data explorer), store-level forecasts (Store forecast page using the trained GRU), ML predictions surfaced directly in the forecast and model-evaluation pages.

---

## 5. Likely Interview Questions + Model Answers

**Q1. Why is a 7-28 day forecast built autoregressively instead of training a
model that directly outputs 28 values at once?**
Direct multi-output regression (predict day+1...day+28 in one forward pass)
avoids compounding error but needs a fixed horizon baked into the architecture
and loses the ability to naturally extend the horizon later. Autoregressive
rollout is simpler to implement correctly and reuses the same single-step
model for any horizon, at the honest cost of compounding error the further out
you forecast — I'd quantify that tradeoff (e.g. report MAPE separately at
day+1, day+7, day+28) before calling this production-ready past a week or two
out.

**Q2. Your holiday-impact SQL query shows holidays have *higher* average
sales, not lower. Isn't that backwards?**
Correctly caught, and worth volunteering: this is Rossmann-specific retail
behavior, not a bug — school/state holidays in this dataset correlate with
stores being open on days they'd normally be closed or quieter (e.g. a
holiday Saturday with extended shopping), and shoppers stock up before/during
holiday periods. It's a reminder that a univariate aggregate can look
counterintuitive until you check *why* — I'd want to control for day-of-week
and promo status jointly before concluding holidays simply "boost" sales
causally.

**Q3. Why MinMaxScaler and not StandardScaler for the sequences?**
RNN-family models (especially with `tanh` activations in vanilla RNN) are
sensitive to input magnitude, and MinMax to [0,1] keeps gradients in a
well-behaved range for recurrent architectures; it's also the more common
default for sequence models feeding into sigmoid/tanh-gated cells like
LSTM/GRU. StandardScaler would work too, but wouldn't bound inputs the way
gated RNN cells are implicitly designed around.

**Q4. Why per-store scalers instead of one global scaler?**
Covered in Section 3, Step 4 — have the store-scale-disparity argument ready
verbatim (flagship vs. small store euro volumes).

**Q5. What would you change for a production version of this system?**
Train on all 1,115 stores (not the 100-store subset used here, which was a
compute-time tradeoff); add a held-out *test* set in addition to the
validation set actually used for model selection (right now, "test" rows in
`test.csv` have no ground truth — they're the Kaggle competition's submission
set, not usable for reporting MAPE); and report per-horizon MAPE (day+1 vs.
day+28) rather than only the single-step number, given the autoregressive
compounding discussed in Q1.

**Q6. Self-critical: what's the weakest part of this project?**
The 100-store training subset (compute-time tradeoff, not a methodology flaw,
but worth naming before being asked "did you use all the stores?") and the
lack of a true held-out test-set MAPE (Kaggle's `test.csv` has no public
labels) — validation MAPE is the honest number to quote, and it's correctly
what's reported throughout this document and the dashboard.

---

## 6. Quick-Reference Numbers (memorize these)

1. **1,017,210** raw rows → **844,338** after cleaning (open stores, sales>0).
2. **GRU: 10.21% MAPE** (158,209 params) — best. LSTM: 12.15% (210,817 params).
   RNN: 15.82% (52,737 params).
3. **30-day lookback**, single-step-ahead target, autoregressive rollout for
   longer horizons.
4. **Chronological per-store 85/15 split** — never random (data leakage).
5. SQL finding to have ready: promo lifts average sales **26-43%** depending
   on store type; holidays show *higher* average sales (counterintuitive,
   explainable — see Q2).
