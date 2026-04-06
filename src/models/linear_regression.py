"""
Target (label_continuous_10d):
    predicted_spread_change = spread(t + 10) - spread(t)
    negative means the spread is expected to contract (mean-revert).
    positive means the spread is expected to widen.

Features (11 columns, pre-computed by pair_dataset_builder.py):
    1. z_score: rolling z-score of OLS spread (60-day lookback)
    2. z_score_kalman: rolling z-score of Kalman spread
    3. momentum_5d: 5-day change in spread
    4. momentum_10d: 10-day change in spread
    5. rolling_vol_20d: 20-day rolling std of daily spread changes
    6. rolling_vol_60d: 60-day rolling std of daily spread changes
    7. rolling_corr_60d: 60-day rolling correlation of the two stocks' returns
    8. days_since_crossing: days since spread last crossed its rolling mean
    9. kalman_beta: current Kalman hedge ratio
    10. kalman_beta_change: 5-day change in Kalman beta
    11. spread_acceleration: second derivative of spread

Evaluation metrics :
    1. MSE 
    2. MAE  
    3. Directional accuracy (% of times sign(predicted) == sign(actual))
    4. Holdout: RMSE (same unit as spread change)

Input:
    data/processed/pair_datasets/<window>/
        train_pair_dataset.csv (required)
        val_pair_dataset.csv (optional (folds 1-4))
        test_pair_dataset.csv (optional (holdout only))

Output:
    data/processed/predictions/linear_regression/<window>/
        predictions.csv  0--> Date, pair, predicted_spread_change

    data/processed/linear_regression_outputs/<window>/pairs/<pair>/
        lr_forecasts_val.csv --> Date, pair, actual, predicted_spread_change, ... (for backtesting)
        lr_forecasts_test.csv --> same
        lr_metrics_val.csv --> mse, mae, rmse, directional_accuracy, n_train, n_eval
        lr_metrics_test.csv --> same

    data/processed/linear_regression_outputs/
        all_val_results.csv --> aggregated val metrics across all windows and pairs
        all_test_results.csv --> aggregated test metrics across all windows and pairs
        fold_summary.csv --> average MSE/MAE per fold (for tuning comparison)
"""

from __future__ import annotations
import re
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import StandardScaler
from src.config import DEFAULT_CONFIG

# features pre-computed by pair_dataset_builder.py 
FEATURE_COLS = [
    "z_score",
    "z_score_kalman",
    "momentum_5d",
    "momentum_10d",
    "rolling_vol_20d",
    "rolling_vol_60d",
    "rolling_corr_60d",
    "days_since_crossing",
    "kalman_beta",
    "kalman_beta_change",
    "spread_acceleration",
]

# target column: spread change over next 10 trading days
TARGET_COL = "label_continuous_10d"
MIN_TRAIN_PTS = 50
MIN_EVAL_PTS  = 10


# HELPER FUNCTIONS
def _safe_pair_name(pair: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.strip())
    return safe.strip("._") or "pair"

def load_pair_dataset(path: Path) -> pd.DataFrame | None:
    if not path.exists():
        return None
    df = pd.read_csv(path, parse_dates = ["Date"])
    return df.dropna(subset = ["Date"]).sort_values(["pair", "Date"]).reset_index(drop = True)

def directional_accuracy(actual: np.ndarray, predicted: np.ndarray) -> float:
    """Percentage of predictions where sign(predicted) == sign(actual).
    Should be above 50% to be better than random guessing.
    """
    mask = actual != 0
    if mask.sum() == 0:
        return float("nan")
    return float((np.sign(actual[mask]) == np.sign(predicted[mask])).mean())

def compute_metrics(actual: np.ndarray, predicted: np.ndarray) -> dict:
    mse  = float(mean_squared_error(actual, predicted))
    mae  = float(mean_absolute_error(actual, predicted))
    rmse = float(np.sqrt(mse))
    da   = directional_accuracy(actual, predicted)
    return {
        "mse": round(mse, 6),
        "mae": round(mae, 6),
        "rmse": round(rmse, 6),
        "directional_accuracy": round(da, 4) if not np.isnan(da) else None,
    }


# LINEAR REGRESSION MODEL
def run_lr_for_pair(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pair: str,
    eval_split: str,
    window_label: str,
) -> tuple[pd.DataFrame, dict] | None:
    """
    Fit a linear regression model on the 11 engineered features and predict
    label_continuous_10d (spread change over next 10 days) on the eval split.
    Returns (forecast_df, metrics) or None if the pair is skipped.
    """

    # extract this pair's rows from train and eval
    train = train_df[train_df["pair"].astype(str) == pair].copy()
    eval_ = eval_df[eval_df["pair"].astype(str) == pair].copy()

    if len(train) < MIN_TRAIN_PTS or len(eval_) < MIN_EVAL_PTS:
        return None

    # check which feature columns are actually available
    available_features = [c for c in FEATURE_COLS if c in train.columns]
    if len(available_features) == 0:
        print(f"[{window_label}:{pair}] No feature columns found.")
        return None
    if len(available_features) < len(FEATURE_COLS):
        missing = set(FEATURE_COLS) - set(available_features)
        print(f"[{window_label}:{pair}] Missing features (will proceed without): {sorted(missing)}")

    if TARGET_COL not in train.columns:
        print(f"[{window_label}:{pair}] Target column '{TARGET_COL}' not found.")
        return None

    # drop rows where any feature or target is NaN
    # also include spread_ols for predicted_value and predicted_z
    extra_cols  = [c for c in ["spread_ols"] if c in eval_.columns]
    train_clean = train[available_features + [TARGET_COL, "Date"]].dropna()
    eval_clean  = eval_[available_features + [TARGET_COL, "Date"] + extra_cols].dropna()

    if len(train_clean) < MIN_TRAIN_PTS or len(eval_clean) < MIN_EVAL_PTS:
        return None

    X_train = train_clean[available_features].values
    y_train = train_clean[TARGET_COL].values
    X_eval  = eval_clean[available_features].values
    y_eval  = eval_clean[TARGET_COL].values

    # standardise, fit only on train to avoid lookahead bias
    scaler    = StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_eval_s  = scaler.transform(X_eval)

    model = LinearRegression()
    model.fit(X_train_s, y_train)
    y_pred = model.predict(X_eval_s)

    metrics = compute_metrics(y_eval, y_pred)
    metrics.update({
        "pair": pair,
        "window_label": window_label,
        "eval_split": eval_split,
        "n_train": len(train_clean),
        "n_eval": len(eval_clean),
        "n_features": len(available_features),
        "target_col": TARGET_COL,
    })

    # predicted_value = current spread + predicted change 
    # predicted_z = predicted change / rolling vol 
    current_spread = eval_clean["spread_ols"].values if "spread_ols" in eval_clean.columns else np.full(len(y_pred), np.nan)
    current_vol = eval_clean["rolling_vol_20d"].values if "rolling_vol_20d" in eval_clean.columns else np.ones(len(y_pred))
    current_vol = np.where(current_vol < 1e-8, 1e-8, current_vol)  # avoid division by zero

    forecast_df = pd.DataFrame({
        "Date": eval_clean["Date"].values,
        "pair": pair,
        "actual_spread_change": y_eval,
        "predicted_spread_change": y_pred,
        "predicted_value": current_spread + y_pred,
        "predicted_z": y_pred / current_vol,
        "forecast_error": y_eval - y_pred,
        "eval_split": eval_split,
        "window_label": window_label,
    })

    return forecast_df, metrics


# FUNCTIONS FOR SAVING
def save_pair_outputs(
    output_root: Path,
    window_label: str,
    pair: str,
    eval_split: str,
    forecast_df: pd.DataFrame,
    metrics: dict,
) -> None:
    pair_dir = output_root / window_label / "pairs" / _safe_pair_name(pair)
    pair_dir.mkdir(parents = True, exist_ok = True)
    forecast_df.to_csv(pair_dir / f"lr_forecasts_{eval_split}.csv", index = False)
    pd.DataFrame([metrics]).to_csv(pair_dir / f"lr_metrics_{eval_split}.csv", index = False)

def save_predictions_for_backtest(
    forecast_df: pd.DataFrame,
    predictions_root: Path,
    window_label: str,
) -> None:
    """Save predictions.csv in the format expected by backtest_engine.py.
    Columns: Date, pair, predicted_spread_change, predicted_value, predicted_z
    """
    pred_dir = predictions_root / window_label
    pred_dir.mkdir(parents = True, exist_ok = True)

    cols = ["Date", "pair", "predicted_spread_change", "predicted_value", "predicted_z"]
    out = forecast_df[[c for c in cols if c in forecast_df.columns]].copy()
    pred_path = pred_dir / "predictions.csv"

    # append if file already exists 
    if pred_path.exists():
        existing = pd.read_csv(pred_path, parse_dates = ["Date"])
        out = pd.concat([existing, out], ignore_index = True).drop_duplicates(
            subset = ["Date", "pair"]).sort_values(["pair", "Date"])

    out.to_csv(pred_path, index = False)

def save_window_outputs(
    output_root: Path,
    window_label: str,
    all_forecasts: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    window_dir = output_root / window_label
    window_dir.mkdir(parents = True, exist_ok = True)
    all_forecasts.to_csv(window_dir / "all_forecasts.csv", index = False)
    metrics_df.to_csv(window_dir / "summary_metrics.csv", index = False)

    for split in sorted(metrics_df["eval_split"].astype(str).unique()):
        all_forecasts[all_forecasts["eval_split"] == split].to_csv(
            window_dir / f"all_forecasts_{split}.csv", index = False)
        metrics_df[metrics_df["eval_split"] == split].to_csv(
            window_dir / f"summary_metrics_{split}.csv", index = False)


# WINDOW RUNNER
def run_window(
    window_dir: Path,
    output_root: Path,
    predictions_root: Path,
    target_pair: str | None = None,
    eval_split_arg: str = "auto",
) -> tuple[list[dict], list[dict]]:
    """Run linear regression for all pairs in one window."""

    window_label = window_dir.name
    train_df = load_pair_dataset(window_dir / "train_pair_dataset.csv")
    val_df   = load_pair_dataset(window_dir / "val_pair_dataset.csv")
    test_df  = load_pair_dataset(window_dir / "test_pair_dataset.csv")

    if train_df is None:
        print(f"[{window_label}] train_pair_dataset.csv not found, skip.")
        return [], []

    # decide which splits to evaluate 
    eval_datasets: list[tuple[str, pd.DataFrame]] = []
    if eval_split_arg in ("val", "auto", "both") and val_df is not None:
        eval_datasets.append(("val", val_df))
    if eval_split_arg in ("test", "both") and test_df is not None:
        eval_datasets.append(("test", test_df))
    if eval_split_arg == "auto" and not eval_datasets and test_df is not None:
        eval_datasets.append(("test", test_df))

    if not eval_datasets:
        print(f"[{window_label}] No eval splits available.")
        return [], []

    all_val_metrics:  list[dict] = []
    all_test_metrics: list[dict] = []
    window_forecasts: list[pd.DataFrame] = []
    window_metrics:   list[dict] = []
    modeled = skipped = 0

    for eval_name, eval_df in eval_datasets:
        train_pairs = set(train_df["pair"].dropna().astype(str).unique())
        eval_pairs  = set(eval_df["pair"].dropna().astype(str).unique())
        pairs = sorted(train_pairs & eval_pairs)

        if target_pair is not None:
            pairs = [target_pair] if target_pair in pairs else []

        print(f"Window: {window_label} | Split: {eval_name} | Pairs: {len(pairs)}")

        for idx, pair in enumerate(pairs, start=1):
            print(f"[{idx}/{len(pairs)}] {pair}")
            try:
                result = run_lr_for_pair(
                    train_df = train_df,
                    eval_df = eval_df,
                    pair = pair,
                    eval_split = eval_name,
                    window_label = window_label,
                )
            except Exception as exc:
                print(f"[{window_label}:{eval_name}:{pair}] Failed: {exc}")
                skipped += 1
                continue

            if result is None:
                skipped += 1
                continue

            forecast_df, metrics = result
            save_pair_outputs(output_root, window_label, pair, eval_name, forecast_df, metrics)

            # save to predictions/ folder for backtest_engine.py
            save_predictions_for_backtest(forecast_df, predictions_root, window_label)

            window_forecasts.append(forecast_df)
            window_metrics.append(metrics)

            if eval_name == "val":
                all_val_metrics.append(metrics)
            else:
                all_test_metrics.append(metrics)

            modeled += 1

    if window_forecasts:
        forecasts_df = pd.concat(window_forecasts, ignore_index = True).sort_values(
            ["eval_split", "pair", "Date"]).reset_index(drop = True)
        metrics_df = pd.DataFrame(window_metrics).sort_values(
            ["eval_split", "mse", "pair"]).reset_index(drop = True)
        save_window_outputs(output_root, window_label, forecasts_df, metrics_df)

    print(f"  Modeled: {modeled} | Skipped: {skipped}")
    return all_val_metrics, all_test_metrics


# MAIN FUNCTION
# to run on a single window or pair, set these — otherwise leave as None to run all
TARGET_WINDOW = None   # e.g. "2010_2012"
TARGET_PAIR = None   # e.g. "gsk-wec"
EVAL_SPLIT = "auto" # "auto", "val", "test", or "both"
                    # set to "test" when running the holdout window

def main() -> None:
    input_root = DEFAULT_CONFIG.processed_dir / "pair_datasets"
    output_root = DEFAULT_CONFIG.processed_dir / "linear_regression_outputs"
    predictions_root = DEFAULT_CONFIG.processed_dir / "predictions" / "linear_regression"

    output_root.mkdir(parents = True, exist_ok = True)
    predictions_root.mkdir(parents = True, exist_ok = True)

    if not input_root.exists():
        raise FileNotFoundError(f"Input root not found: {input_root}")

    window_dirs = sorted([p for p in input_root.iterdir() if p.is_dir()], key = lambda p: p.name)
    if TARGET_WINDOW:
        window_dirs = [d for d in window_dirs if d.name == TARGET_WINDOW]
        if not window_dirs:
            raise ValueError(f"Window '{TARGET_WINDOW}' not found under {input_root}")

    all_val_results: list[dict] = []
    all_test_results: list[dict] = []

    for window_dir in window_dirs:
        print(f"\nWindow {window_dir.name.replace('_', '-')}")
        val_rows, test_rows = run_window(
            window_dir = window_dir,
            output_root = output_root,
            predictions_root = predictions_root,
            target_pair = TARGET_PAIR,
            eval_split_arg = EVAL_SPLIT,
        )
        all_val_results.extend(val_rows)
        all_test_results.extend(test_rows)

    # save aggregated results
    if all_val_results:
        val_agg = pd.DataFrame(all_val_results).sort_values("mse")
        val_agg.to_csv(output_root / "all_val_results.csv", index = False)
        print(f"\nSaved: {output_root / 'all_val_results.csv'}")

    if all_test_results:
        test_agg = pd.DataFrame(all_test_results).sort_values("mse")
        test_agg.to_csv(output_root / "all_test_results.csv", index = False)
        print(f"Saved: {output_root / 'all_test_results.csv'}")

    # fold summary: average val MSE/MAE per window (used for tuning comparison table)
    if all_val_results:
        fold_summary = (
            pd.DataFrame(all_val_results)
            .groupby("window_label")[["mse", "mae", "rmse", "directional_accuracy"]]
            .mean()
            .round(6)
            .reset_index()
        )
        fold_summary.to_csv(output_root / "fold_summary.csv", index = False)
        print(f"Saved: {output_root / 'fold_summary.csv'}")
        print("\nFold summary (avg val metrics per window):")
        print(fold_summary.to_string(index = False))

    print("\nLinear Regression Complete")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Predictions root: {predictions_root}")
    print(f"Target: {TARGET_COL}")
    print(f"Features: {len(FEATURE_COLS)}")
    print(f"Val pairs: {len(all_val_results)}")
    print(f"Test pairs: {len(all_test_results)}")


if __name__ == "__main__":
    main()