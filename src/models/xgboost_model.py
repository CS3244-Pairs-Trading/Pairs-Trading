"""
XGBoost Spread Change Pipeline
CS3244 Machine Learning - Group 23

Runs both OLS and Kalman XGBoost variants end-to-end per MODEL_BRIEF.

Why this is one file:
    XGBoost is a global model: all pairs pooled into one matrix, 18 fits total
    per window. The full pipeline runs in seconds, so splitting it across files
    only adds indirection with no benefit.

Pipeline (per MODEL_BRIEF):
    Step 1 — Tune across all rolling windows
             Run all 18 hyperparameter combos on each window's val set.
             Average val MSE across windows per combo → pick globally best params.

    Step 2 — Train + predict on each rolling window
             Retrain with best params on each window's train set.
             Generate predictions on each window's val set.

    Step 3 — Holdout evaluation
             Retrain with best params on holdout train set.
             Generate predictions on holdout test set.

Output (per MODEL_BRIEF):
    data/processed/predictions/xgboost_ols/<window>/predictions.csv
    data/processed/predictions/xgboost_kalman/<window>/predictions.csv

Prediction CSV columns:
    Date, pair, predicted_change, predicted_value, predicted_z

Tuning summary saved to:
    data/processed/predictions/xgboost_ols/tuning_summary.csv
    data/processed/predictions/xgboost_kalman/tuning_summary.csv

Usage:
    python -m src.models.xgboost_model
    python -m src.models.xgboost_model --spread_type ols
    python -m src.models.xgboost_model --no_tune
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG
from .xgboost_spread_model import (
    FEATURE_NAMES,
    PARAM_GRID,
    TARGET_KALMAN,
    TARGET_KALMAN_ZSCORE,
    TARGET_OLS,
    TARGET_OLS_ZSCORE,
    SpreadChangeXGBoost,
    XGBoostPipeline,
    evaluate_predictions,
)

warnings.filterwarnings("ignore")


def _resolve_target(spread_type: str, use_zscore: bool) -> str:
    """Return the correct target column name for the given spread type and z-score flag."""
    if use_zscore:
        return TARGET_OLS_ZSCORE if spread_type == "ols" else TARGET_KALMAN_ZSCORE
    return TARGET_OLS if spread_type == "ols" else TARGET_KALMAN


# ─────────────────────────────────────────────────────────────────────────────
# Window discovery
# ─────────────────────────────────────────────────────────────────────────────

def iter_window_dirs(root: Path) -> List[Path]:
    """Return sorted rolling window dirs, ignoring sibling folders."""
    if not root.exists():
        raise FileNotFoundError(f"Pair datasets root not found: {root}")
    dirs = sorted(
        [p for p in root.iterdir() if p.is_dir()],
        key=lambda p: p.name,
    )
    if not dirs:
        raise ValueError(f"No 'window_*' directories found in {root}")
    return dirs


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def _load_csv(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    return df.dropna(subset=["Date"]).sort_values("Date").reset_index(drop=True)


def load_window_splits(
    window_dir: Path,
    eval_split: str = "val",
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Load train + eval CSVs for one window.

    Args:
        window_dir : path to one window folder
        eval_split : 'val' for rolling windows, 'test' for holdout

    Returns:
        (train_df, eval_df) — eval_df is None if file absent
    """
    train_path = window_dir / "train_pair_dataset.csv"
    eval_path  = window_dir / f"{eval_split}_pair_dataset.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split: {train_path}")

    train_df = _load_csv(train_path)
    eval_df  = _load_csv(eval_path) if eval_path.exists() else None
    return train_df, eval_df


# ─────────────────────────────────────────────────────────────────────────────
# Column validation
# ─────────────────────────────────────────────────────────────────────────────

def validate_columns(
    df: pd.DataFrame,
    spread_type: str,
    window_label: str = "",
) -> None:
    """Raise a clear error if any required columns are missing."""
    target_col = TARGET_OLS   if spread_type == "ols" else TARGET_KALMAN
    spread_col = "spread_ols" if spread_type == "ols" else "spread_kalman"
    required   = set(FEATURE_NAMES) | {target_col, spread_col, "Date", "pair"}
    missing    = required - set(df.columns)
    if missing:
        tag = f"[{window_label}] " if window_label else ""
        raise ValueError(
            f"{tag}Missing columns: {sorted(missing)}\n"
            f"These must be produced by pair_dataset_builder upstream."
        )


# ─────────────────────────────────────────────────────────────────────────────
# Feature / label extraction
# ─────────────────────────────────────────────────────────────────────────────

def _extract_pair_Xy(
    df: pd.DataFrame,
    pair: str,
    target_col: str,
) -> Optional[Tuple[np.ndarray, np.ndarray]]:
    """Return (X, y) for one pair, dropping NaN rows. None if empty."""
    pair_df = df[df["pair"].astype(str) == str(pair)].sort_values("Date")
    clean   = pair_df.dropna(subset=FEATURE_NAMES + [target_col])
    if clean.empty:
        return None
    return (
        clean[FEATURE_NAMES].values.astype(float),
        clean[target_col].values.astype(float),
    )


def _stack_window(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    target_col: str,
    spread_col: str,
    min_samples: int,
) -> Tuple[
    Dict[str, np.ndarray], Dict[str, np.ndarray],
    Dict[str, np.ndarray], Dict[str, np.ndarray],
]:
    """
    Build {pair → X/y} dicts for all shared pairs in one window.
    Pairs with fewer than min_samples rows in either split are dropped.
    """
    shared = sorted(
        set(train_df["pair"].dropna().astype(str).unique())
        & set(eval_df["pair"].dropna().astype(str).unique())
    )
    X_tr, y_tr, X_ev, y_ev = {}, {}, {}, {}
    for pair in shared:
        tr = _extract_pair_Xy(train_df, pair, target_col)
        ev = _extract_pair_Xy(eval_df,  pair, target_col)
        if tr is None or ev is None:
            continue
        if len(tr[1]) < min_samples or len(ev[1]) < min_samples:
            continue
        X_tr[pair], y_tr[pair] = tr
        X_ev[pair], y_ev[pair] = ev
    return X_tr, y_tr, X_ev, y_ev


# ─────────────────────────────────────────────────────────────────────────────
# Prediction builder
# ─────────────────────────────────────────────────────────────────────────────

def _build_predictions(
    df: pd.DataFrame,
    pair: str,
    model: SpreadChangeXGBoost,
    target_col: str,
    spread_col: str,
) -> Optional[pd.DataFrame]:
    """
    Run model on one pair, return the 5-column output per MODEL_BRIEF.
    actual_change is included temporarily for eval; stripped before saving.
    """
    pair_df = df[df["pair"].astype(str) == str(pair)].sort_values("Date").copy()
    clean   = pair_df.dropna(subset=FEATURE_NAMES)
    if clean.empty:
        return None

    X                = clean[FEATURE_NAMES].values.astype(float)
    predicted_change = model.predict(X)
    current_spread   = clean[spread_col].values.astype(float)
    rolling_vol_20d  = clean["rolling_vol_20d"].values.astype(float)

    _, predicted_value, predicted_z = SpreadChangeXGBoost.derive_outputs(
        predicted_change, current_spread, rolling_vol_20d
    )
    return pd.DataFrame({
        "Date":             clean["Date"].values,
        "pair":             clean["pair"].values,
        "predicted_change": predicted_change,
        "predicted_value":  predicted_value,
        "predicted_z":      predicted_z,
        "actual_change":    clean[target_col].values.astype(float),
    })


def _collect_predictions(
    eval_df: pd.DataFrame,
    pairs: List[str],
    model: SpreadChangeXGBoost,
    target_col: str,
    spread_col: str,
) -> pd.DataFrame:
    """Collect and concatenate predictions for all pairs in a split."""
    rows = [
        _build_predictions(eval_df, pair, model, target_col, spread_col)
        for pair in pairs
    ]
    rows = [r for r in rows if r is not None]
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# Saving
# ─────────────────────────────────────────────────────────────────────────────

def _save_predictions(
    predictions_df: pd.DataFrame,
    metrics: Dict,
    out_dir: Path,
) -> None:
    """Save predictions.csv (5 cols) and metrics.csv to out_dir."""
    out_dir.mkdir(parents=True, exist_ok=True)
    save_cols = ["Date", "pair", "predicted_change", "predicted_value", "predicted_z"]
    predictions_df[save_cols].to_csv(out_dir / "predictions.csv", index=False)
    pd.DataFrame([metrics]).to_csv(out_dir / "metrics.csv", index=False)


# ─────────────────────────────────────────────────────────────────────────────
# Step 1 — Tune across all rolling windows
# ─────────────────────────────────────────────────────────────────────────────

def tune_across_windows(
    window_dirs: List[Path],
    spread_type: str,
    min_samples: int,
    use_zscore: bool = False,
) -> Tuple[Dict, pd.DataFrame]:
    """
    Run all 18 hyperparameter combos on every rolling window, then select
    the globally best combo by composite score (R² + directional accuracy).

    Returns:
        best_params : {"max_depth", "n_estimators", "learning_rate"}
        tuning_df   : full grid results (n_windows × 18 rows)
    """
    target_col = _resolve_target(spread_type, use_zscore)
    spread_col = "spread_ols" if spread_type == "ols" else "spread_kalman"
    combos     = list(itertools.product(
        PARAM_GRID["max_depth"],
        PARAM_GRID["n_estimators"],
        PARAM_GRID["learning_rate"],
    ))

    print(f"\n── Tuning ({len(combos)} combos × {len(window_dirs)} windows) ──")

    all_rows: List[Dict] = []

    for window_dir in window_dirs:
        window_label = window_dir.name
        try:
            train_df, val_df = load_window_splits(window_dir, eval_split="val")
        except FileNotFoundError:
            continue
        if val_df is None:
            continue
        try:
            validate_columns(train_df, spread_type, window_label)
        except ValueError as e:
            warnings.warn(str(e), stacklevel=2)
            continue

        X_tr_d, y_tr_d, X_vl_d, y_vl_d = _stack_window(
            train_df, val_df, target_col, spread_col, min_samples
        )
        if not X_tr_d:
            continue

        pipeline = XGBoostPipeline(spread_type=spread_type)
        X_train, y_train = pipeline.stack_pairs(X_tr_d, y_tr_d)
        X_val,   y_val   = pipeline.stack_pairs(X_vl_d, y_vl_d)

        for depth, n_est, lr in combos:
            m = SpreadChangeXGBoost(max_depth=depth, n_estimators=n_est, learning_rate=lr)
            m.fit(X_train, y_train, X_val, y_val, verbose=False)
            metrics = evaluate_predictions(y_val, m.predict(X_val))
            all_rows.append({
                "window_label":  window_label,
                "max_depth":     depth,
                "n_estimators":  n_est,
                "learning_rate": lr,
                "val_rmse":      metrics["rmse"],
                "val_directional_weighted_mse": metrics["directional_weighted_mse"],
                "val_dir_acc":   metrics["directional_accuracy"],
                "val_r2":        metrics["r2"],
                "val_ic":        metrics["information_coefficient"],
                "val_pw_da":     metrics["profit_weighted_da"],
            })

    if not all_rows:
        raise ValueError("No tuning results produced — check pair dataset columns.")

    tuning_df = pd.DataFrame(all_rows)

    # Average metrics across windows per combo
    avg = (
        tuning_df
        .groupby(["max_depth", "n_estimators", "learning_rate"], as_index=False)
        .agg(
            mean_val_rmse   =("val_rmse",    "mean"),
            mean_val_directional_weighted_mse=("val_directional_weighted_mse", "mean"),
            mean_val_dir_acc=("val_dir_acc", "mean"),
            mean_val_r2     =("val_r2",      "mean"),
            mean_val_ic     =("val_ic",      "mean"),
            mean_val_pw_da  =("val_pw_da",   "mean"),
            n_windows       =("window_label","nunique"),
        )
    )

    # Composite selection score: 0.5 * R² + 0.5 * directional_accuracy
    # Both are on [0, 1] scale (R² can be negative but that's fine — it
    # correctly penalises models worse than predicting the mean).
    avg["composite_score"] = 0.5 * avg["mean_val_r2"] + 0.5 * avg["mean_val_dir_acc"]
    avg = avg.sort_values(
        ["composite_score", "mean_val_directional_weighted_mse", "mean_val_rmse"],
        ascending=[False, True, True],
    ).reset_index(drop=True)

    best = avg.iloc[0]
    best_params = {
        "max_depth":        int(best["max_depth"]),
        "n_estimators":     int(best["n_estimators"]),
        "learning_rate":    float(best["learning_rate"]),
    }

    print(f"\n  {'depth':>5}  {'n_est':>5}  {'lr':>5}  {'avg_mse':>10}  {'avg_dw':>10}  {'avg_r2':>8}"
          f"  {'avg_da':>7}  {'composite':>10}")
    print(f"  {'-'*78}")
    for _, row in avg.iterrows():
        marker = " ←" if (
            int(row["max_depth"])     == best_params["max_depth"] and
            int(row["n_estimators"])  == best_params["n_estimators"] and
            float(row["learning_rate"]) == best_params["learning_rate"]
        ) else ""
        print(
            f"  {int(row['max_depth']):>5}  {int(row['n_estimators']):>5}  "
            f"{row['learning_rate']:>5.2f}  {row['mean_val_rmse']:>10.6f}  "
            f"{row['mean_val_directional_weighted_mse']:>10.6f}  "
            f"{row['mean_val_r2']:>8.4f}  {row['mean_val_dir_acc']:>7.3f}  "
            f"{row['composite_score']:>10.4f}{marker}"
        )

    print(f"\n  Best params  : depth={best_params['max_depth']}  "
          f"n_est={best_params['n_estimators']}  lr={best_params['learning_rate']}")
    print(f"  Avg val MSE  : {float(best['mean_val_rmse']):.6f}")
    print(f"  Avg val DW-MSE: {float(best['mean_val_directional_weighted_mse']):.6f}")
    print(f"  Avg val R²   : {float(best['mean_val_r2']):.4f}")
    print(f"  Avg val IC   : {float(best['mean_val_ic']):.4f}")
    print(f"  Avg dir acc  : {float(best['mean_val_dir_acc']):.3f}")
    print(f"  Composite    : {float(best['composite_score']):.4f}")
    print(f"  Windows used : {int(best['n_windows'])}")

    return best_params, tuning_df


# ─────────────────────────────────────────────────────────────────────────────
# Step 2 — Train + predict on each rolling window
# ─────────────────────────────────────────────────────────────────────────────

def run_rolling_windows(
    window_dirs: List[Path],
    spread_type: str,
    best_params: Dict,
    preds_root: Path,
    min_samples: int,
    use_zscore: bool = False,
) -> Dict[str, Dict]:
    """
    With best_params fixed, retrain on each window's train set and
    generate predictions on the val set.

    Returns per-window metrics dict.
    """
    target_col = _resolve_target(spread_type, use_zscore)
    spread_col = "spread_ols" if spread_type == "ols" else "spread_kalman"
    model_name = f"xgboost_{spread_type}"

    print(f"\n── Rolling window predictions ──")
    results: Dict[str, Dict] = {}

    for window_dir in window_dirs:
        window_label = window_dir.name
        try:
            train_df, val_df = load_window_splits(window_dir, eval_split="val")
        except FileNotFoundError:
            continue
        if val_df is None:
            continue

        X_tr_d, y_tr_d, X_vl_d, y_vl_d = _stack_window(
            train_df, val_df, target_col, spread_col, min_samples
        )
        if not X_tr_d:
            continue

        pipeline = XGBoostPipeline(spread_type=spread_type)
        X_train, y_train = pipeline.stack_pairs(X_tr_d, y_tr_d)
        X_val,   y_val   = pipeline.stack_pairs(X_vl_d, y_vl_d)

        pipeline.train(X_train, y_train, X_val, y_val, **best_params)
        metrics = pipeline.evaluate(X_val, y_val)

        predictions_df = _collect_predictions(
            val_df, sorted(X_vl_d.keys()),
            pipeline.best_model, target_col, spread_col,
        )

        metrics_row = {
            "window_label": window_label, "spread_type": spread_type,
            "model": model_name, "n_pairs": len(X_vl_d),
            "n_val_samples": len(X_val),
            "val_rmse": metrics["rmse"],
            "val_directional_weighted_mse": metrics["directional_weighted_mse"],
            "val_dir_acc": metrics["directional_accuracy"],
            "val_r2": metrics["r2"],
            "val_ic": metrics["information_coefficient"],
            "val_pw_da": metrics["profit_weighted_da"],
            **{f"best_{k}": v for k, v in best_params.items()},
        }

        _save_predictions(
            predictions_df, metrics_row,
            preds_root / model_name / window_label,
        )
        results[window_label] = {
            "metrics": metrics, "n_pairs": len(X_vl_d),
            "n_val_samples": len(X_val),
        }
        print(
            f"  [{window_label}]  MSE={metrics['mse']:.6f}  DW-MSE={metrics['directional_weighted_mse']:.6f}  R²={metrics['r2']:.4f}  "
            f"IC={metrics['information_coefficient']:.4f}  DirAcc={metrics['directional_accuracy']:.3f}  "
            f"({len(X_vl_d)} pairs)"
        )

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Step 3 — Holdout evaluation
# ─────────────────────────────────────────────────────────────────────────────

def run_holdout(
    holdout_window: str,
    spread_type: str,
    best_params: Dict,
    datasets_root: Path,
    preds_root: Path,
    min_samples: int,
    use_zscore: bool = False,
) -> Optional[Dict]:
    """
    Retrain with best_params on the holdout train set,
    generate predictions on the holdout test set.

    The holdout window folder must contain test_pair_dataset.csv.
    """
    target_col = _resolve_target(spread_type, use_zscore)
    spread_col = "spread_ols" if spread_type == "ols" else "spread_kalman"
    model_name = f"xgboost_{spread_type}"

    window_dir = datasets_root / holdout_window
    try:
        train_df, test_df = load_window_splits(window_dir, eval_split="test")
    except FileNotFoundError as e:
        warnings.warn(str(e), stacklevel=2)
        return None

    if test_df is None:
        warnings.warn(
            f"[{holdout_window}] test_pair_dataset.csv not found — skipping holdout.",
            stacklevel=2,
        )
        return None

    validate_columns(train_df, spread_type, holdout_window)

    X_tr_d, y_tr_d, X_te_d, y_te_d = _stack_window(
        train_df, test_df, target_col, spread_col, min_samples
    )
    if not X_tr_d:
        warnings.warn(f"[{holdout_window}] No pairs with sufficient data.", stacklevel=2)
        return None

    pipeline = XGBoostPipeline(spread_type=spread_type)
    X_train, y_train = pipeline.stack_pairs(X_tr_d, y_tr_d)
    X_test,  y_test  = pipeline.stack_pairs(X_te_d, y_te_d)

    pipeline.train(X_train, y_train, **best_params)
    metrics = pipeline.evaluate(X_test, y_test)

    predictions_df = _collect_predictions(
        test_df, sorted(X_te_d.keys()),
        pipeline.best_model, target_col, spread_col,
    )

    metrics_row = {
        "window_label": holdout_window, "spread_type": spread_type,
        "model": model_name, "split": "test",
        "n_pairs": len(X_te_d), "n_test_samples": len(X_test),
        "test_rmse": metrics["rmse"],
        "test_directional_weighted_mse": metrics["directional_weighted_mse"],
        "test_dir_acc": metrics["directional_accuracy"],
        "test_r2": metrics["r2"],
        "test_ic": metrics["information_coefficient"],
        "test_pw_da": metrics["profit_weighted_da"],
        **{f"best_{k}": v for k, v in best_params.items()},
    }

    _save_predictions(
        predictions_df, metrics_row,
        preds_root / model_name / holdout_window,
    )

    print(
        f"\n── Holdout [{holdout_window}] ──\n"
        f"  MSE={metrics['mse']:.6f}  DW-MSE={metrics['directional_weighted_mse']:.6f}  R²={metrics['r2']:.4f}  "
        f"IC={metrics['information_coefficient']:.4f}  DirAcc={metrics['directional_accuracy']:.3f}  "
        f"({len(X_te_d)} pairs, {len(X_test):,} rows)"
    )
    return {"metrics": metrics, "n_pairs": len(X_te_d)}


# ─────────────────────────────────────────────────────────────────────────────
# Top-level runner — one spread variant end-to-end
# ─────────────────────────────────────────────────────────────────────────────

def run_variant(
    spread_type: str = "ols",
    datasets_root: Optional[Path] = None,
    preds_root: Optional[Path] = None,
    min_samples: int = 30,
    tune: bool = True,
    target_window: Optional[str] = None,
    use_zscore: bool = False,
) -> Dict:
    """
    Full end-to-end pipeline for one spread variant.

    Args:
        spread_type    : 'ols' or 'kalman'
        datasets_root  : override DEFAULT_CONFIG path
        preds_root     : override DEFAULT_CONFIG path
        min_samples    : minimum rows per pair per split
        tune           : if False, use default params (skip grid search)
        target_window  : run only this one window (useful for debugging)

    Returns:
        {
          "best_params"   : dict,
          "tuning_df"     : DataFrame of all grid results,
          "window_results": {window_label: metrics dict},
          "holdout"       : metrics dict or None,
        }
    """
    datasets_root = datasets_root or (DEFAULT_CONFIG.processed_dir / "pair_datasets")
    preds_root    = preds_root    or (DEFAULT_CONFIG.processed_dir / "predictions")
    target_suffix = "_zscore" if use_zscore else ""
    model_name    = f"xgboost_{spread_type}{target_suffix}"

    print(f"\n{'='*60}")
    print(f"XGBoost {spread_type.upper()} variant{' (z-scored target)' if use_zscore else ''}")
    print(f"{'='*60}")

    # Discover rolling windows (excludes holdout)
    all_window_dirs = iter_window_dirs(datasets_root)
    holdout_label   = DEFAULT_CONFIG.holdout_split.label
    rolling_dirs    = [d for d in all_window_dirs if d.name != holdout_label]

    if target_window is not None:
        rolling_dirs = [d for d in rolling_dirs if d.name == target_window]
        if not rolling_dirs:
            raise ValueError(f"Window '{target_window}' not found (or it is the holdout window).")

    # ── Step 1: Tune across rolling windows ───────────────────────────
    if tune:
        best_params, tuning_df = tune_across_windows(
            rolling_dirs, spread_type, min_samples, use_zscore=use_zscore
        )
    else:
        best_params = {
            "max_depth": args.max_depth if args.max_depth is not None else 4,
            "n_estimators": args.n_estimators if args.n_estimators is not None else 200,
            "learning_rate": args.learning_rate if args.learning_rate is not None else 0.05,
        }
        tuning_df   = pd.DataFrame()
        print(f"\n  Skipping tuning — using default params: {best_params}")

    # Save tuning summary
    out_base = preds_root / model_name
    out_base.mkdir(parents=True, exist_ok=True)
    if not tuning_df.empty:
        tuning_df.to_csv(out_base / "tuning_summary.csv", index=False)

    pd.DataFrame([{**best_params, "spread_type": spread_type}]).to_csv(
        out_base / "best_params.csv", index=False
    )

    # ── Step 2: Train + predict on rolling windows ────────────────────
    window_results = run_rolling_windows(
        rolling_dirs, spread_type, best_params, preds_root, min_samples,
        use_zscore=use_zscore,
    )

    # ── Step 3: Holdout evaluation ────────────────────────────────────
    holdout_result = None
    if target_window is None:   # skip holdout when debugging a single window
        holdout_result = run_holdout(
            holdout_label, spread_type, best_params,
            datasets_root, preds_root, min_samples,
            use_zscore=use_zscore,
        )

    return {
        "best_params":    best_params,
        "tuning_df":      tuning_df,
        "window_results": window_results,
        "holdout":        holdout_result,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Run both variants + comparison table
# ─────────────────────────────────────────────────────────────────────────────

def run_both_variants(
    datasets_root: Optional[Path] = None,
    preds_root: Optional[Path] = None,
    min_samples: int = 30,
    tune: bool = True,
    target_window: Optional[str] = None,
    use_zscore: bool = False,
) -> Dict[str, Dict]:
    """
    Run xgboost_ols and xgboost_kalman end-to-end, then print a comparison.

    OLS vs Kalman comparison is on avg val R² (within-model).
    Cross-type Sharpe comparison happens in the backtest engine.
    """
    ols_result    = run_variant("ols",    datasets_root, preds_root, min_samples, tune, target_window, use_zscore)
    kalman_result = run_variant("kalman", datasets_root, preds_root, min_samples, tune, target_window, use_zscore)

    # Summary table
    windows = sorted(set(ols_result["window_results"]) | set(kalman_result["window_results"]))
    print(f"\n{'='*60}")
    print("OLS vs Kalman — val RMSE per window")
    print(f"  {'Window':<22}  {'OLS RMSE':>10}  {'Kalman RMSE':>12}  {'Better':>8}")
    print(f"  {'-'*57}")
    for w in windows:
        ols_rmse    = ols_result["window_results"].get(w,    {}).get("metrics", {}).get("rmse", float("nan"))
        kalman_rmse = kalman_result["window_results"].get(w, {}).get("metrics", {}).get("rmse", float("nan"))
        better     = "OLS" if ols_rmse <= kalman_rmse else "Kalman"
        print(f"  {w:<22}  {ols_rmse:>10.6f}  {kalman_rmse:>12.6f}  {better:>8}")

    for label, res in [("OLS", ols_result), ("Kalman", kalman_result)]:
        if res["holdout"]:
            m = res["holdout"]["metrics"]
            print(
                f"\n  Holdout {label}: RMSE={m['rmse']:.6f}  DW-MSE={m['directional_weighted_mse']:.6f}  R²={m['r2']:.4f}  "
                f"IC={m['information_coefficient']:.4f}  DirAcc={m['directional_accuracy']:.3f}"
            )

    return {"ols": ols_result, "kalman": kalman_result}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run XGBoost OLS and/or Kalman spread change pipeline."
    )
    parser.add_argument(
        "--spread_type", choices=["ols", "kalman", "both"], default="both",
    )
    parser.add_argument(
        "--pair_datasets_root", type=str, default=None,
        help="Override DEFAULT_CONFIG path for pair_datasets/",
    )
    parser.add_argument(
        "--predictions_root", type=str, default=None,
        help="Override DEFAULT_CONFIG path for predictions/",
    )
    parser.add_argument(
        "--window", type=str, default=None,
        help="Debug mode: run only this rolling window (holdout skipped).",
    )
    parser.add_argument(
        "--no_tune", action="store_true",
        help="Skip grid search and use default hyperparameters.",
    )
    parser.add_argument(
        "--zscore_target", action="store_true",
        help="Train on z-scored targets (spread_change / rolling_vol_20d) "
             "instead of raw spread change. Makes MSE naturally calibrated.",
    )
    parser.add_argument("--min_samples", type=int, default=30)
    parser.add_argument("--max_depth", type=int, default=None)
    parser.add_argument("--n_estimators", type=int, default=None)
    parser.add_argument("--learning_rate", type=float, default=None)
    args = parser.parse_args()

    datasets_root = Path(args.pair_datasets_root) if args.pair_datasets_root else None
    preds_root    = Path(args.predictions_root)    if args.predictions_root    else None

    if args.spread_type == "both":
        run_both_variants(datasets_root, preds_root, args.min_samples,
                          not args.no_tune, args.window)
    else:
        run_variant(args.spread_type, datasets_root, preds_root,
                    args.min_samples, not args.no_tune, args.window)
