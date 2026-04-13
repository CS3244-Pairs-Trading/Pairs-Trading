"""
Kalman Filter for dynamic hedge ratios.
========================================
Replaces the static OLS beta from rank_pairs.py with a time-varying
hedge ratio that adapts as the relationship between two stocks shifts.

The state is the hedge ratio (beta). The observation model is:
    price_A(t) = beta(t) * price_B(t) + noise

The Kalman Filter estimates beta(t) at each time step, producing a
time-varying hedge ratio and a dynamic spread:
    spread(t) = price_A(t) - beta(t) * price_B(t)

Usage
-----
    from src.pairs_discovery.kalman_hedge import (
        kalman_hedge_ratio,
        kalman_spread,
        kalman_vs_ols_comparison,
    )

    # Get time-varying hedge ratio for one pair
    beta_series = kalman_hedge_ratio(log_prices_a, log_prices_b)

    # Get the spread constructed with Kalman beta
    spread, beta = kalman_spread(log_prices_a, log_prices_b)

    # Compare Kalman vs OLS for all discovered pairs
    comparison_df = kalman_vs_ols_comparison(prices_pivot, pairs_df)

Integration with pipeline
-------------------------
This module is called by feature_engineering.py to add Kalman-based
features (kalman_spread, kalman_beta, kalman_beta_change) alongside
the existing OLS-based spread features.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def kalman_hedge_ratio(
    price_a: np.ndarray | pd.Series,
    price_b: np.ndarray | pd.Series,
    delta: float = 1e-4,
    obs_noise: float = 1.0,
) -> np.ndarray:
    """
    Estimate a time-varying hedge ratio using a 2D Kalman Filter.

    State vector: [beta, intercept]
        state(t) = state(t-1) + w(t),   w ~ N(0, delta * I)

    Observation model:
        price_a(t) = beta(t) * price_b(t) + intercept(t) + v(t)

    This matches the OLS model in rank_pairs.py (np.polyfit with deg=1)
    which fits both slope and intercept. Without the intercept term,
    the filter absorbs any level difference into beta, biasing it.

    Parameters
    ----------
    price_a : array-like
        Log prices of stock A (the dependent variable).
    price_b : array-like
        Log prices of stock B (the independent variable).
    delta : float
        Process noise variance — controls how fast the state is allowed
        to change. Smaller = smoother. 1e-4 to 1e-5 is typical for
        daily log prices.
    obs_noise : float
        Initial observation noise variance (R).

    Returns
    -------
    beta : np.ndarray of shape (n,)
        Time-varying hedge ratio at each time step (the slope component).
    """
    a = np.asarray(price_a, dtype=np.float64)
    b = np.asarray(price_b, dtype=np.float64)
    n = len(a)

    # state: [beta, intercept] — 2D
    state = np.zeros((n, 2))
    P = np.eye(2)            # state covariance (2x2)
    R = obs_noise            # observation noise (scalar)
    Q = delta * np.eye(2)    # process noise (2x2)

    # initialise with OLS on first 20 observations
    warmup = min(20, n)
    if warmup >= 2:
        coeffs = np.polyfit(b[:warmup], a[:warmup], deg=1)
        state[0] = [coeffs[0], coeffs[1]]   # [slope, intercept]
    else:
        state[0] = [1.0, 0.0]

    for t in range(1, n):
        # predict
        state_prior = state[t - 1]          # [beta, intercept]
        P_prior = P + Q

        # observation matrix: a(t) = [price_b(t), 1] . [beta, intercept]
        H = np.array([b[t], 1.0])           # (2,)
        y_hat = H @ state_prior              # predicted observation
        innovation = a[t] - y_hat            # prediction error

        # innovation covariance (scalar)
        S = H @ P_prior @ H + R

        # Kalman gain (2,)
        K = P_prior @ H / S

        # update
        state[t] = state_prior + K * innovation
        P = P_prior - np.outer(K, H) @ P_prior

    return state[:, 0]   # return only beta (slope), not intercept


def kalman_spread(
    price_a: np.ndarray | pd.Series,
    price_b: np.ndarray | pd.Series,
    delta: float = 1e-4,
    obs_noise: float = 1.0,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Construct a dynamic spread using the Kalman-filtered hedge ratio.

    spread(t) = price_a(t) - beta(t) * price_b(t)

    Parameters
    ----------
    price_a, price_b : array-like
        Log prices of the two stocks.
    delta, obs_noise : float
        Kalman Filter tuning parameters (see kalman_hedge_ratio).

    Returns
    -------
    spread : np.ndarray
        Dynamic spread at each time step.
    beta : np.ndarray
        Time-varying hedge ratio at each time step.
    """
    a = np.asarray(price_a, dtype=np.float64)
    b = np.asarray(price_b, dtype=np.float64)

    beta = kalman_hedge_ratio(a, b, delta=delta, obs_noise=obs_noise)
    spread = a - beta * b

    return spread, beta


def kalman_vs_ols_comparison(
    prices_pivot: pd.DataFrame,
    pairs_df: pd.DataFrame,
    delta: float = 1e-4,
) -> pd.DataFrame:
    """
    Compare Kalman vs OLS hedge ratios for all discovered pairs.

    For each pair, computes:
    - OLS beta (static, from rank_pairs.py's initial_beta)
    - Kalman beta (final value)
    - Spread stationarity (ADF p-value) for both OLS and Kalman spreads
    - Spread std for both

    Parameters
    ----------
    prices_pivot : pd.DataFrame
        Wide-format prices with Date index and Ticker columns.
    pairs_df : pd.DataFrame
        Output from rank_pairs.py with columns: pair, initial_beta, is_eligible.
    delta : float
        Kalman process noise parameter.

    Returns
    -------
    pd.DataFrame with comparison metrics for each pair.
    """
    from statsmodels.tsa.stattools import adfuller

    results = []
    log_prices = np.log(prices_pivot)

    for _, row in pairs_df.iterrows():
        if not row.get("is_eligible", False):
            continue

        pair_name = row["pair"]
        s1 = row["stock_a"]
        s2 = row["stock_b"]

        if s1 not in log_prices.columns or s2 not in log_prices.columns:
            continue

        p1 = log_prices[s1].dropna()
        p2 = log_prices[s2].dropna()
        common_idx = p1.index.intersection(p2.index)
        if len(common_idx) < 60:
            continue

        p1 = p1.loc[common_idx].values
        p2 = p2.loc[common_idx].values

        # OLS spread
        ols_beta = float(row["initial_beta"])
        ols_spread = p1 - ols_beta * p2

        # Kalman spread
        k_spread, k_beta = kalman_spread(p1, p2, delta=delta)

        # ADF tests
        try:
            ols_adf_pval = adfuller(ols_spread, maxlag=20, autolag="AIC")[1]
        except Exception:
            ols_adf_pval = np.nan

        try:
            kalman_adf_pval = adfuller(k_spread[20:], maxlag=20, autolag="AIC")[1]
        except Exception:
            kalman_adf_pval = np.nan

        results.append({
            "pair": pair_name,
            "ols_beta": ols_beta,
            "kalman_beta_final": k_beta[-1],
            "kalman_beta_std": np.std(k_beta[20:]),
            "ols_spread_std": np.std(ols_spread),
            "kalman_spread_std": np.std(k_spread[20:]),
            "ols_adf_pval": ols_adf_pval,
            "kalman_adf_pval": kalman_adf_pval,
            "kalman_more_stationary": kalman_adf_pval < ols_adf_pval
            if not (np.isnan(kalman_adf_pval) or np.isnan(ols_adf_pval))
            else None,
        })

    return pd.DataFrame(results)
