"""
Feature engineering for spread models.
=======================================
Builds the feature matrix (X) and labels (y) consumed by all downstream
models (logistic regression, XGBoost, ARMA, LSTM).

For each (pair, date) it computes features from the spread time series.
Both OLS-based and Kalman-based spread features are included.

Input
-----
- prices_features.csv  (long format: Date, Ticker, Close, SimpleReturn, LogPrice)
- discovered_pairs.csv (output of rank_pairs.py)

Output
------
- Per training window: a CSV with columns:
    Date, pair, stock_a, stock_b,
    spread_ols, spread_kalman, z_score, z_score_kalman,
    momentum_5d, momentum_10d, rolling_vol_20d, rolling_vol_60d,
    rolling_corr_60d, rel_volume_a, rel_volume_b,
    days_since_crossing, kalman_beta, kalman_beta_change,
    spread_acceleration,
    label_binary_5d, label_binary_10d, label_continuous_5d, label_continuous_10d

Usage
-----
    python -m src.models.feature_engineering
    python -m src.models.feature_engineering --window 2010_2012 --horizon 10

Integration
-----------
    from src.models.feature_engineering import (
        build_features_for_window,
        build_all_features,
    )
    X_train, y_train, X_val, y_val = build_features_for_window(
        full_df, pairs_df, fold, volume_df=vol_df
    )
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import (
    DEFAULT_CONFIG,
    ExpandingFold,
    all_training_windows,
)
from src.pairs_discovery.kalman_hedge import kalman_spread


# ---------------------------------------------------------------------------
# SPREAD CONSTRUCTION
# ---------------------------------------------------------------------------

def compute_ols_spread(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    beta: float,
) -> pd.Series:
    """
    Static OLS spread: spread(t) = log_price_a(t) - beta * log_price_b(t).

    Uses the initial_beta from rank_pairs.py (frozen from training period).
    """
    return log_price_a - beta * log_price_b


def compute_kalman_spread(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    delta: float = 1e-4,
) -> tuple[pd.Series, pd.Series]:
    """
    Dynamic Kalman spread + time-varying beta.

    Returns (spread_series, beta_series) with the same index as inputs.
    """
    common_idx = log_price_a.index.intersection(log_price_b.index)
    a = log_price_a.loc[common_idx]
    b = log_price_b.loc[common_idx]

    spread_arr, beta_arr = kalman_spread(a.values, b.values, delta=delta)

    return (
        pd.Series(spread_arr, index=common_idx, name="spread_kalman"),
        pd.Series(beta_arr, index=common_idx, name="kalman_beta"),
    )


# ---------------------------------------------------------------------------
# FEATURE COMPUTATION (per pair)
# ---------------------------------------------------------------------------

def _z_score(spread: pd.Series, lookback: int = 60) -> pd.Series:
    """Rolling Z-score of the spread."""
    mu = spread.rolling(lookback, min_periods=max(1, lookback // 2)).mean()
    sigma = spread.rolling(lookback, min_periods=max(1, lookback // 2)).std().clip(lower=1e-8)
    return (spread - mu) / sigma


def _momentum(spread: pd.Series, window: int) -> pd.Series:
    """Spread change over the last `window` days."""
    return spread.diff(window)


def _rolling_vol(spread: pd.Series, window: int) -> pd.Series:
    """Rolling std of daily spread changes."""
    return spread.diff().rolling(window, min_periods=max(1, window // 2)).std()


def _rolling_corr(ret_a: pd.Series, ret_b: pd.Series, window: int = 60) -> pd.Series:
    """Rolling correlation of the two stocks' returns."""
    return ret_a.rolling(window, min_periods=max(1, window // 2)).corr(ret_b)


def _relative_volume(volume: pd.Series, window: int = 20) -> pd.Series:
    """Volume relative to its rolling average."""
    avg = volume.rolling(window, min_periods=max(1, window // 2)).mean().clip(lower=1.0)
    return volume / avg


def _days_since_crossing(spread: pd.Series) -> pd.Series:
    """Number of days since the spread last crossed its rolling mean."""
    demeaned = spread - spread.rolling(60, min_periods=30).mean()
    sign_change = (demeaned.shift(1) * demeaned) < 0
    groups = sign_change.cumsum()
    return groups.groupby(groups).cumcount()


def _spread_acceleration(spread: pd.Series, window: int = 5) -> pd.Series:
    """Second derivative of the spread — is the spread change accelerating?"""
    velocity = spread.diff(window)
    return velocity.diff(window)


def compute_pair_features(
    log_price_a: pd.Series,
    log_price_b: pd.Series,
    return_a: pd.Series,
    return_b: pd.Series,
    ols_beta: float,
    volume_a: pd.Series | None = None,
    volume_b: pd.Series | None = None,
    kalman_delta: float = 1e-4,
) -> pd.DataFrame:
    """
    Compute the full feature set for one pair.

    All features use only past data (rolling windows) — no future leakage.

    Parameters
    ----------
    log_price_a, log_price_b : pd.Series
        Log prices of the two stocks (DatetimeIndex).
    return_a, return_b : pd.Series
        Simple daily returns of the two stocks.
    ols_beta : float
        Static hedge ratio from rank_pairs.py (initial_beta).
    volume_a, volume_b : pd.Series, optional
        Daily volume for relative volume features. Skipped if None.
    kalman_delta : float
        Kalman Filter process noise parameter.

    Returns
    -------
    pd.DataFrame
        One row per trading day, columns = feature names.
    """
    common_idx = (
        log_price_a.index
        .intersection(log_price_b.index)
        .intersection(return_a.index)
        .intersection(return_b.index)
    )
    lp_a = log_price_a.loc[common_idx]
    lp_b = log_price_b.loc[common_idx]
    r_a = return_a.loc[common_idx]
    r_b = return_b.loc[common_idx]

    # OLS spread
    spread_ols = compute_ols_spread(lp_a, lp_b, ols_beta)

    # Kalman spread + dynamic beta
    spread_kalman, kalman_beta = compute_kalman_spread(lp_a, lp_b, delta=kalman_delta)

    features = pd.DataFrame(index=common_idx)

    # Spread values (for reference, not necessarily used as features)
    features["spread_ols"] = spread_ols
    features["spread_kalman"] = spread_kalman

    # Z-scores (the core trading signals)
    features["z_score"] = _z_score(spread_ols, lookback=60)
    features["z_score_kalman"] = _z_score(spread_kalman, lookback=60)

    # Momentum features
    features["momentum_5d"] = _momentum(spread_ols, 5)
    features["momentum_10d"] = _momentum(spread_ols, 10)

    # Volatility features
    features["rolling_vol_20d"] = _rolling_vol(spread_ols, 20)
    features["rolling_vol_60d"] = _rolling_vol(spread_ols, 60)

    # Correlation between the two stocks
    features["rolling_corr_60d"] = _rolling_corr(r_a, r_b, 60)

    # Volume features (optional)
    if volume_a is not None and volume_b is not None:
        vol_a = volume_a.reindex(common_idx)
        vol_b = volume_b.reindex(common_idx)
        features["rel_volume_a"] = _relative_volume(vol_a, 20)
        features["rel_volume_b"] = _relative_volume(vol_b, 20)

    # Days since mean crossing
    features["days_since_crossing"] = _days_since_crossing(spread_ols)

    # Kalman-specific features
    features["kalman_beta"] = kalman_beta
    features["kalman_beta_change"] = kalman_beta.diff(5)

    # Second-order feature
    features["spread_acceleration"] = _spread_acceleration(spread_ols, 5)

    return features


# ---------------------------------------------------------------------------
# LABELS
# ---------------------------------------------------------------------------

def compute_labels(
    lp_a: pd.Series,
    lp_b: pd.Series,
    current_beta: pd.Series,
    ols_beta: float,
    horizons: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Construct labels for spread prediction.

    Binary label: 1 if |spread| moved closer to the mean over the next N days.
    Continuous label: the actual spread change over the next N days (for regression).
    Kalman labels: same but using the Kalman spread as target.

    The labels use FUTURE data — they are targets, not features.
    They must only be used in the training set to avoid leakage.
    """
    labels = pd.DataFrame(index=lp_a.index)
    spread_ols = lp_a - ols_beta * lp_b
    rolling_mean_ols = spread_ols.rolling(60, min_periods=30).mean()
    demeaned_ols = spread_ols - rolling_mean_ols

    for h in horizons:
        future_spread_ols = spread_ols.shift(-h)
        # Binary: did spread get closer to rolling mean?
        # Continuous: OLS spread change (for regression models)
        labels[f"label_continuous_{h}d"] = future_spread_ols - spread_ols
        labels[f"label_binary_{h}d"] = ((future_spread_ols - rolling_mean_ols.shift(-h)).abs() < demeaned_ols.abs()).astype(float)

        # Kalman spread change (for Kalman variant comparison)
        current_spread_k = lp_a - (current_beta * lp_b)
        # Future value at t+h using Beta from time t
        future_val_locked = lp_a.shift(-h) - (current_beta * lp_b.shift(-h))
        labels[f"label_kalman_{h}d"] = future_val_locked - current_spread_k

    return labels


# ---------------------------------------------------------------------------
# WINDOW-LEVEL FEATURE BUILDING
# ---------------------------------------------------------------------------

def build_features_for_pairs(
    full_df: pd.DataFrame,
    pairs_df: pd.DataFrame,
    train_end: str,
    val_start: str | None = None,
    val_end: str | None = None,
    kalman_delta: float = 1e-4,
    label_horizons: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """
    Build the complete feature matrix for all eligible pairs in one window.

    Parameters
    ----------
    full_df : pd.DataFrame
        Long-format prices with columns: Date, Ticker, Close, Volume,
        SimpleReturn, LogPrice. (prices_features.csv)
    pairs_df : pd.DataFrame
        Filtered pairs for this training window (is_eligible == True).
        Must have columns: pair, initial_beta.
    train_end : str
        Last date of the training period (features are computed up to here,
        plus the validation period if val_start/val_end are provided).
    val_start, val_end : str, optional
        Validation period boundaries. If None, only training features
        are returned.
    kalman_delta : float
        Kalman Filter tuning parameter.
    label_horizons : tuple of int
        Forward-looking horizons for label construction.

    Returns
    -------
    pd.DataFrame
        Stacked features for all pairs with columns:
        Date, pair, stock_a, stock_b, [all features], [all labels]
    """
    eligible = pairs_df[pairs_df["is_eligible"].astype(bool)].copy()
    if eligible.empty:
        return pd.DataFrame()

    # Determine date range for feature computation
    # We need the full range (train + val) so rolling features warm up properly
    if val_end is not None:
        date_mask = full_df["Date"] <= pd.Timestamp(val_end)
    else:
        date_mask = full_df["Date"] <= pd.Timestamp(train_end)
    window_df = full_df[date_mask].copy()

    # Pivot to wide format
    prices_wide = window_df.pivot_table(
        index="Date", columns="Ticker", values="Close", aggfunc="last"
    ).sort_index()

    log_prices_wide = window_df.pivot_table(
        index="Date", columns="Ticker", values="LogPrice", aggfunc="last"
    ).sort_index()

    returns_wide = window_df.pivot_table(
        index="Date", columns="Ticker", values="SimpleReturn", aggfunc="last"
    ).sort_index()

    # Volume (optional — may not exist in all datasets)
    has_volume = "Volume" in window_df.columns
    if has_volume:
        volume_wide = window_df.pivot_table(
            index="Date", columns="Ticker", values="Volume", aggfunc="last"
        ).sort_index()

    all_pair_features = []

    for _, row in eligible.iterrows():
        pair_name = row["pair"]
        s1, s2 = pair_name.split("-", 1)
        ols_beta = float(row["initial_beta"])

        # Check both tickers exist
        if s1 not in log_prices_wide.columns or s2 not in log_prices_wide.columns:
            continue
        if s1 not in returns_wide.columns or s2 not in returns_wide.columns:
            continue

        lp_a = log_prices_wide[s1].dropna()
        lp_b = log_prices_wide[s2].dropna()
        r_a = returns_wide[s1].dropna()
        r_b = returns_wide[s2].dropna()

        vol_a = volume_wide[s1].dropna() if has_volume and s1 in volume_wide.columns else None
        vol_b = volume_wide[s2].dropna() if has_volume and s2 in volume_wide.columns else None

        # Compute features
        feat = compute_pair_features(
            lp_a, lp_b, r_a, r_b, ols_beta,
            volume_a=vol_a, volume_b=vol_b,
            kalman_delta=kalman_delta,
        )

        # Compute labels (using OLS spread as the target)
        spread_for_labels = compute_ols_spread(lp_a, lp_b, ols_beta)
        labs = compute_labels(spread_for_labels.reindex(feat.index), horizons=label_horizons)

        # Merge features + labels
        pair_df = pd.concat([feat, labs], axis=1)
        pair_df["pair"] = pair_name
        pair_df["stock_a"] = s1
        pair_df["stock_b"] = s2
        pair_df.index.name = "Date"

        all_pair_features.append(pair_df.reset_index())

    if not all_pair_features:
        return pd.DataFrame()

    result = pd.concat(all_pair_features, ignore_index=True)
    return result


def split_train_val(
    features_df: pd.DataFrame,
    train_end: str,
    val_start: str,
    val_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split the feature matrix into train and validation sets.

    Drops rows with NaN in any feature column (from rolling warmup period).
    """
    feature_cols = [
        c for c in features_df.columns
        if c not in ("Date", "pair", "stock_a", "stock_b")
        and not c.startswith("label_")
        and not c.startswith("spread_")
    ]
    label_cols = [c for c in features_df.columns if c.startswith("label_")]

    all_needed = feature_cols + label_cols
    clean = features_df.dropna(subset=all_needed)

    train_mask = clean["Date"] <= pd.Timestamp(train_end)
    val_mask = (
        (clean["Date"] >= pd.Timestamp(val_start))
        & (clean["Date"] <= pd.Timestamp(val_end))
    )

    return clean[train_mask].copy(), clean[val_mask].copy()


def get_feature_columns() -> list[str]:
    """Return the list of feature column names used by models."""
    return [
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
        # Volume features are optional — included if present
        # "rel_volume_a",
        # "rel_volume_b",
    ]


# ---------------------------------------------------------------------------
# MAIN: build features for all windows and save
# ---------------------------------------------------------------------------

def build_all_features(
    full_df: pd.DataFrame | None = None,
    pairs_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    kalman_delta: float = 1e-4,
) -> None:
    """
    Build and save feature matrices for all expanding windows.

    Output files:
        <output_dir>/<window_label>/features.csv
        <output_dir>/<window_label>/train_features.csv
        <output_dir>/<window_label>/val_features.csv
    """
    config = DEFAULT_CONFIG

    if full_df is None:
        print(f"Loading prices from: {config.engineered_features_path}")
        full_df = pd.read_csv(config.engineered_features_path, parse_dates=["Date"])

    if pairs_df is None:
        pairs_path = config.processed_dir / "discovered_pairs.csv"
        print(f"Loading pairs from: {pairs_path}")
        pairs_df = pd.read_csv(pairs_path)

    if output_dir is None:
        output_dir = config.processed_dir / "features"

    # Process each expanding fold
    for fold in config.expanding_folds:
        print(f"\n{'='*60}")
        print(f"Window: {fold.label}")
        print(f"  Train: {fold.train.start} – {fold.train.end}")
        print(f"  Val:   {fold.val.start} – {fold.val.end}")

        # Filter pairs for this window
        window_pairs = pairs_df[pairs_df["training_window"] == fold.label]
        n_eligible = window_pairs["is_eligible"].sum() if not window_pairs.empty else 0
        print(f"  Pairs: {len(window_pairs)} total, {n_eligible} eligible")

        if n_eligible == 0:
            print("  [SKIP] No eligible pairs")
            continue

        # Build features
        feat = build_features_for_pairs(
            full_df=full_df,
            pairs_df=window_pairs,
            train_end=fold.train.end,
            val_start=fold.val.start,
            val_end=fold.val.end,
            kalman_delta=kalman_delta,
        )

        if feat.empty:
            print("  [SKIP] No features produced")
            continue

        # Split and save
        train_feat, val_feat = split_train_val(
            feat, fold.train.end, fold.val.start, fold.val.end
        )

        window_out = output_dir / fold.label
        window_out.mkdir(parents=True, exist_ok=True)

        feat.to_csv(window_out / "features_all.csv", index=False)
        train_feat.to_csv(window_out / "train_features.csv", index=False)
        val_feat.to_csv(window_out / "val_features.csv", index=False)

        print(f"  Saved: {len(train_feat)} train rows, {len(val_feat)} val rows")
        print(f"  → {window_out}")

    # Also build for holdout
    hs = config.holdout_split
    print(f"\n{'='*60}")
    print(f"Holdout: {hs.label}")
    print(f"  Train: {hs.train.start} – {hs.train.end}")
    print(f"  Test:  {hs.test.start} – {hs.test.end}")

    holdout_pairs = pairs_df[pairs_df["training_window"] == hs.label]
    n_eligible = holdout_pairs["is_eligible"].sum() if not holdout_pairs.empty else 0

    if n_eligible > 0:
        feat = build_features_for_pairs(
            full_df=full_df,
            pairs_df=holdout_pairs,
            train_end=hs.train.end,
            val_start=hs.test.start,
            val_end=hs.test.end,
            kalman_delta=kalman_delta,
        )

        if not feat.empty:
            train_feat, test_feat = split_train_val(
                feat, hs.train.end, hs.test.start, hs.test.end
            )

            holdout_out = output_dir / hs.label
            holdout_out.mkdir(parents=True, exist_ok=True)

            feat.to_csv(holdout_out / "features_all.csv", index=False)
            train_feat.to_csv(holdout_out / "train_features.csv", index=False)
            test_feat.to_csv(holdout_out / "test_features.csv", index=False)

            print(f"  Saved: {len(train_feat)} train rows, {len(test_feat)} test rows")
            print(f"  → {holdout_out}")

    print(f"\n{'='*60}")
    print("Done.")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Build spread features for all windows")
    parser.add_argument("--window", type=str, default=None, help="Run only this window label")
    parser.add_argument("--horizon", type=int, nargs="+", default=[5, 10], help="Label horizons")
    parser.add_argument("--delta", type=float, default=1e-4, help="Kalman delta parameter")
    args = parser.parse_args()

    build_all_features(kalman_delta=args.delta)
