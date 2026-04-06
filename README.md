# CS3244 Group Project: Pairs Trading with Machine Learning

Exploring pairs trading strategies to identify profitable stock relationships using ML-enhanced spread prediction.

---

## Quick start

```bash
git clone <repo-url>
cd Pairs-Trading
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

### Data setup

Download [Price Volume Data for All US Stocks & ETFs](https://www.kaggle.com/datasets/borismarjanovic/price-volume-data-for-all-us-stocks-etfs) from Kaggle, then:

```bash
mkdir -p data/raw
cp /path/to/downloaded/Stocks/*.txt data/raw/
```

---

## Pipeline overview

```
Raw stock data
  → prepare_data.py .............. clean, filter top 1000, compute returns
  → pairs_discovery.py .......... PCA, clustering, cointegration, rank pairs
  → pairs_selection.py .......... select top K pairs per window
  → pair_dataset_builder.py ..... build train/val CSVs with features + labels
  → models (ARMA, lin reg, XGB, LSTM) .. predict spread change
  → trading strategy ............ convert predictions to signals
  → backtest_engine.py .......... simulate trades, compute Sharpe/drawdown
```

---

## Sliding window setup

We use a 3-year sliding window. Old data drops off so pair selection and model
training reflect the current market regime rather than stale relationships.

| Fold | Train period | Validate on | Label |
|------|-------------|-------------|-------|
| 1 | 2010–2012 | 2013 | `2010_2012` |
| 2 | 2011–2013 | 2014 | `2011_2013` |
| 3 | 2012–2014 | 2015 | `2012_2014` |
| 4 | 2013–2015 | 2016 | `2013_2015` |
| **Holdout** | **2014–2016** | **2017** | `2014_2016` |

**Purpose of the 4 folds:** tune hyperparameters by averaging validation MSE across all 4 folds. Pick the settings with the lowest average. Then freeze those settings and run the holdout (2017) exactly once.

**Top-K pairs differ per window.** Fold 1 discovers pairs from 2010–2012 data. Fold 2 discovers different pairs from 2011–2013 data. The same pair may or may not appear across folds. This is correct — relationships change over time.

---

## Running the full pipeline (step by step)

### Step 1: Data preparation

```bash
python3 prepare_data.py
```

**What it does:** Filters top 1,000 stocks by dollar volume, cleans prices, handles outliers, computes SimpleReturn and LogPrice.

**Output:** `data/processed/prices_features.csv` (long format, ~1.7M rows)

**Columns:** `Date, Ticker, Open, High, Low, Close, Volume, SimpleReturn, LogPrice`

### Step 2: Pair discovery

```bash
python3 pairs_discovery.py
```

Or run each stage separately:

```bash
python3 -m src.clustering.pca          # PCA per window
python3 -m src.clustering.optics       # OPTICS clustering per window
python3 -m src.pairs_discovery.rank_pairs   # cointegration testing + scoring
```

**Output:** `data/processed/discovered_pairs.csv`

### Step 3: Select top K pairs

```bash
python3 -m src.pairs_discovery.pairs_selection --top_k 20
```

**Output:** `data/processed/selected_pairs/<window_label>/selected_pairs.csv`

### Step 4: Build pair datasets (features + labels for models)

```bash
python3 -m src.models.pair_dataset_builder
```

**Output per window:**

```
data/processed/pair_datasets/
├── 2010_2012/
│   ├── pair_dataset.csv          ← full (train + val)
│   ├── train_pair_dataset.csv    ← train split only
│   └── val_pair_dataset.csv      ← validation split only
├── 2011_2013/ ...
├── 2012_2014/ ...
├── 2013_2015/ ...
└── 2014_2016/                    ← holdout
    ├── train_pair_dataset.csv
    └── test_pair_dataset.csv     ← "test" not "val" for holdout
```

**Key columns in pair dataset CSVs:**

```
Metadata:     Date, pair, stock_a, stock_b, training_window
Raw prices:   close_a, close_b, return_a, return_b, log_price_a, log_price_b, volume_a, volume_b
Spreads:      spread_ols, spread_kalman
Features:     z_score, z_score_kalman, momentum_5d, momentum_10d,
              rolling_vol_20d, rolling_vol_60d, rolling_corr_60d,
              days_since_crossing, kalman_beta, kalman_beta_change, spread_acceleration
Labels:       label_binary_5d, label_binary_10d, label_continuous_5d, label_continuous_10d,
              label_kalman_5d, label_kalman_10d
```

### Step 5: Run models (see "Models" section below)

### Step 6: Run backtest

```bash
python3 -m src.backtest.backtest_engine              # validation folds
python3 -m src.backtest.backtest_engine --holdout     # final test on 2017
```

---

## Models — standardised specification

### What every model predicts

All models predict the same target: **`label_continuous_10d`** — the change in the OLS spread over the next 10 trading days.

```
label_continuous_10d = spread(t + 10) - spread(t)
```

where `spread = log(price_A) - beta * log(price_B)`.

A negative prediction means the spread is expected to contract (mean-revert). A positive prediction means it is expected to widen.

### Unified prediction output format

Every model must save predictions in this format so the trading strategy layer can consume them interchangeably:

```
data/processed/predictions/<model_name>/<window_label>/predictions.csv

Columns: Date, pair, predicted_spread_change
```

### Feature columns (input to lin reg, XGBoost, LSTM)

```python
feature_cols = [
    'z_score',             # rolling z-score of OLS spread (60-day lookback)
    'z_score_kalman',      # rolling z-score of Kalman spread
    'momentum_5d',         # 5-day change in spread
    'momentum_10d',        # 10-day change in spread
    'rolling_vol_20d',     # 20-day rolling std of daily spread changes
    'rolling_vol_60d',     # 60-day rolling std of daily spread changes
    'rolling_corr_60d',    # 60-day rolling correlation of the two stocks' returns
    'days_since_crossing', # days since spread last crossed its rolling mean
    'kalman_beta',         # current Kalman hedge ratio
    'kalman_beta_change',  # 5-day change in Kalman beta
    'spread_acceleration', # second derivative of spread
]
```

### Model details

| Model | Input | How it works | Owner |
|-------|-------|-------------|-------|
| **OU baseline** | Raw spread only | Fits Ornstein-Uhlenbeck parameters (kappa, theta, sigma) on a rolling lookback window. Produces z-scores from OU equilibrium — no features, no training. | David |
| **ARMA** | Raw spread only | ARMA via `ARIMA(order=(p,0,q))` on raw spread values. Predicts 10-day-ahead spread value first, then derives predicted change and predicted z-score at forecast origin. Supports both OLS and Kalman spread variants. | Priscilla |
| **Linear regression** | 11 features | `predicted_change = w0 + w1*z_score + w2*momentum + ...` Simplest feature-based model. | Mabel |
| **XGBoost** | 11 features | Gradient-boosted trees. Captures non-linear interactions (e.g. high z-score + low volatility = strong reversion signal). | Catherine |
| **LSTM** | 20-day sequences of 11 features | Sees temporal patterns — how features evolved over the last 20 days. One sample = (20, 11) matrix → one prediction. | Kenneth/Priscilla |

### Running each model

```bash
# OU baseline
python3 -m src.models.ou_extended

# ARMA (fixed params, OLS variant)
python3 -m src.models.arma \
  --spread_col spread_ols \
  --p 3 --q 2 \
  --horizon 10 \
  --eval_split val

# ARMA (fixed params, Kalman variant)
python3 -m src.models.arma \
  --spread_col spread_kalman \
  --p 3 --q 2 \
  --horizon 10 \
  --eval_split val

# ARMA global tuning (12x12) - OLS
python3 -m src.models.arma_tuning \
  --spread_col spread_ols \
  --horizon 10

# ARMA global tuning (12x12) - Kalman
python3 -m src.models.arma_tuning \
  --spread_col spread_kalman \
  --horizon 10

# ARMA holdout evaluation using selected global params - OLS
python3 -m src.models.arma_holdout_eval \
  --spread_col spread_ols \
  --horizon 10 \
  --save_forecasts

# ARMA holdout evaluation using selected global params - Kalman
python3 -m src.models.arma_holdout_eval \
  --spread_col spread_kalman \
  --horizon 10 \
  --save_forecasts

# Linear regression, XGBoost, LSTM
python3 -m src.models.linear_regression
python3 -m src.models.xgboost_model
python3 -m src.models.lstm
```

### ARMA pipeline (current behavior)

- ARMA is fit per pair.
- Two variants are supported:
  - `spread_ols`
  - `spread_kalman`
- Forecast horizon is 10 trading days by default.
- ARMA predicts future spread value first (`predicted_value`), then derives:
  - `predicted_change = predicted_value - current_spread`
  - `predicted_z = predicted_change / rolling_vol_20d_at_origin`
- ARMA files do not generate trading signals and do not run backtesting metrics.
- Layer-1 metrics in ARMA files are `mse` and `mae` on predicted change.

### ARMA outputs (where files go)

**Fixed-parameter ARMA runs**

```
data/processed/predictions/
├── arma_ols/
│   └── <window_label>/
│       ├── all_forecasts.csv
│       ├── all_forecasts_val.csv / all_forecasts_test.csv
│       ├── summary_metrics.csv
│       ├── summary_metrics_val.csv / summary_metrics_test.csv
│       └── pairs/<pair>/arma_forecasts_<split>.csv
└── arma_kalman/
    └── <window_label>/...
```

Per-pair ARMA forecast CSVs include:
`forecast_origin_date, target_date, pair, spread_col, predicted_change, predicted_value, predicted_z, actual_change, actual_value`

**ARMA tuning outputs**

```
data/processed/arma_tuning_outputs/
├── arma_ols/
│   ├── all_validation_results.csv
│   ├── global_param_ranking.csv
│   └── selected_global_params.csv
└── arma_kalman/
    ├── all_validation_results.csv
    ├── global_param_ranking.csv
    └── selected_global_params.csv
```

Tuning is global per spread type (not per pair): one selected `(p,q)` for OLS and one selected `(p,q)` for Kalman.

**ARMA holdout outputs**

```
data/processed/arma_holdout_outputs/
├── arma_ols/
│   ├── final_holdout_results.csv
│   ├── holdout_summary.csv
│   ├── selected_global_params_used.csv
│   └── <window_label>/pairs/<pair>/arma_forecasts_test.csv   # if --save_forecasts
└── arma_kalman/
    └── ...
```

### Hyperparameter tuning process

Tune by averaging validation MSE across all 4 sliding folds. Pick the settings with the lowest average. Freeze those settings for the holdout.

```python
for params in hyperparameter_grid:
    fold_mses = []
    for fold in all_4_folds:
        model.fit(fold.X_train, fold.y_train)
        preds = model.predict(fold.X_val)
        fold_mses.append(mean_squared_error(fold.y_val, preds))
    avg_mse = np.mean(fold_mses)

# Pick params with lowest avg_mse → freeze for holdout
```

Hyperparameter grids per model:

| Model | Hyperparameters to tune |
|-------|------------------------|
| ARMA | p: [1..12], q: [1..12] (12x12 global grid search, selected by validation MSE then MAE) |
| Linear regression | None (no hyperparameters) |
| XGBoost | max_depth: [3,4,5], n_estimators: [100,200], learning_rate: [0.01,0.05,0.1] |
| LSTM | hidden_size: [32,64], window_size: [10,20], learning_rate: [0.001,0.01] |

---

## Evaluation metrics

### Model evaluation (does the model predict spread change well?)

| Metric | Formula | Used for |
|--------|---------|----------|
| **MSE** | mean((predicted - actual)^2) | Primary metric for all regression models |
| **MAE** | mean(\|predicted - actual\|) | Secondary — less sensitive to outliers |
| **RMSE** | sqrt(MSE) | Same unit as the spread change (Only used in the holdout set)|
| **Directional accuracy** | % of times sign(predicted) == sign(actual) | Sanity check — should be above 50% |

These are computed on validation data per fold, then averaged across all 4 folds.

### Trading evaluation (does the strategy make money?)

| Metric | What it measures |
|--------|-----------------|
| **Sharpe ratio** | Risk-adjusted return: (annualized_return - risk_free) / volatility. Above 1.0 is good. |
| **Max drawdown** | Worst peak-to-trough decline. How much you would lose at the worst point. |
| **Annualized return** | Yearly return if you ran the strategy for a full year. |
| **Volatility** | Annualized standard deviation of daily returns. |
| **Number of trades** | Must be above 30 for statistical significance. |
| **Turnover** | How much capital is traded annually. High turnover = high transaction costs. |
| **Fitness** | Sharpe × sqrt(\|return\| / max(turnover, 12.5%)). Penalizes high-turnover strategies. |

### What the report comparison table should look like

```
| Model    | Avg Val MSE | Avg Val MAE | Avg Val Sharpe | Avg Val MaxDD |
|----------|-------------|-------------|----------------|---------------|
| OU       | ...         | ...         | ...            | ...           |
| ARMA     | ...         | ...         | ...            | ...           |
| Lin Reg  | ...         | ...         | ...            | ...           |
| XGBoost  | ...         | ...         | ...            | ...           |
| LSTM     | ...         | ...         | ...            | ...           |
```

Each row uses the best-tuned version of that model (best hyperparameters selected via 4-fold validation average). All evaluated on the same validation pairs and dates.

---

## After changing config.py (switching to sliding windows)

If you update `config.py` from expanding to sliding windows, you must rerun the pipeline from step 2 onward to regenerate pair discovery and datasets with the new window dates:

```bash
# Step 1 NOT needed — prices_features.csv covers full 2010-2017 range

# Step 2: Rerun pair discovery with new windows
python3 pairs_discovery.py

# Step 3: Reselect top K pairs
python3 -m src.pairs_discovery.pairs_selection --top_k 20

# Step 4: Rebuild pair datasets
python3 -m src.models.pair_dataset_builder

# Step 5: Retrain all models on new pair datasets
```

---

## Project structure

```
├── src/
│   ├── config.py                          # Sliding window dates, paths, shared config
│   ├── pipeline.py                        # Data loading orchestrator
│   ├── data_prep/
│   │   ├── data_cleaning.py               # Raw CSV → long-format DataFrame
│   │   ├── filter_stocks.py               # Top 1000 by dollar volume
│   │   ├── isolate_top_1000.py            # Copy selected stock files
│   │   ├── returns.py                     # Adds SimpleReturn + LogPrice
│   │   ├── splits.py                      # Train/val date filtering helpers
│   │   ├── handle_outliers.py             # Outlier detection and forward-fill
│   │   └── feature_engineering.py         # Spread features (OLS + Kalman)
│   ├── clustering/
│   │   ├── pca.py                         # PCA on return matrix per window
│   │   ├── kmeans.py                      # K-Means clustering
│   │   ├── hierarchical.py                # Agglomerative clustering (Ward)
│   │   ├── dbscan.py                      # DBSCAN clustering
│   │   └── optics.py                      # OPTICS clustering (default)
│   ├── pairs_discovery/
│   │   ├── rank_pairs.py                  # Cointegration testing + pair scoring
│   │   ├── pairs_selection.py             # Filter top K pairs per window
│   │   └── kalman_hedge.py                # 2D Kalman filter for dynamic hedge ratios
│   ├── models/
│   │   ├── pair_dataset_builder.py        # Builds train/val CSVs with features + labels
│   │   ├── feature_engineering.py         # Feature computation module
│   │   ├── arma.py                        # ARMA spread forecasting
│   │   ├── arma_tuning.py                 # ARMA hyperparameter grid search
│   │   ├── arma_holdout_eval.py           # ARMA final holdout test
│   │   ├── ou.py                          # OU baseline model
│   │   └── ou_extended.py                 # OU + GARCH + regime-switching + VECM
│   │   └── lstm.py                        # LSTM spread prediction + tuning
│   └── backtest/
│       └── backtest_engine.py             # Pluggable signal-based backtester
├── tests/
│   └── test_kalman.py                     # Kalman filter validation tests
├── prepare_data.py                        # Full data prep pipeline (run first)
├── pairs_discovery.py                     # Full pair discovery pipeline
├── requirements.txt
└── data/                                  # gitignored — regenerate with pipeline
```

---

## Team

| Member | Role |
|--------|------|
| Priscilla Ashley Wijaya | Project leader, ARMA, LSTM, pipeline architect |
| Catherine Anne Listijo | XGBoost, SHAP analysis |
| David Michael Indraputra | OU baseline, walk-forward orchestration |
| Felix Tanwira | Backtest engine, trading strategy |
| Kenneth Christopher Hendra | Kalman filter, feature engineering, LSTM |
| Mabel Augustine Anggoro | PCA, clustering, linear regression, sensitivity analysis |

Mentor: Junchuan
