"""
Test script for kalman_hedge.py
================================
Run from project root:
    python tests/test_kalman.py

Tests:
1. Synthetic cointegrated pair (known beta) — does Kalman converge?
2. Synthetic regime-change pair (beta shifts) — does Kalman adapt?
3. Real pairs from discovered_pairs.csv — compare Kalman vs OLS stationarity
"""

import sys
from pathlib import Path

# Add project root to path so we can import src.*
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.pairs_discovery.kalman_hedge import (
    kalman_hedge_ratio,
    kalman_spread,
    kalman_vs_ols_comparison,
)


def test_1_synthetic_stable_beta():
    """
    Test: two stocks with a known, constant beta = 1.5.
    The Kalman filter should converge to ~1.5 after warmup.
    """
    print("=" * 60)
    print("TEST 1: Synthetic pair with stable beta = 1.5")
    print("=" * 60)

    np.random.seed(42)
    n = 500

    # Stock B follows a random walk
    log_b = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0

    # Stock A = 1.5 * stock B + mean-reverting noise
    true_beta = 1.5
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + np.random.normal(0, 0.005)

    log_a = true_beta * log_b + 2.0 + noise

    # Run Kalman
    beta = kalman_hedge_ratio(log_a, log_b, delta=1e-4)

    # Check convergence (after 50-day warmup)
    beta_final_avg = np.mean(beta[-100:])
    beta_std = np.std(beta[-100:])

    print(f"  True beta:          {true_beta:.4f}")
    print(f"  Kalman beta (avg):  {beta_final_avg:.4f}")
    print(f"  Kalman beta (std):  {beta_std:.4f}")
    print(f"  Error:              {abs(beta_final_avg - true_beta):.4f}")

    assert abs(beta_final_avg - true_beta) < 0.1, \
        f"Kalman beta {beta_final_avg:.4f} too far from true {true_beta}"
    assert beta_std < 0.05, \
        f"Kalman beta too unstable: std = {beta_std:.4f}"

    print("  PASSED\n")


def test_2_regime_change():
    """
    Test: beta shifts from 1.0 to 2.0 halfway through.
    The Kalman filter should adapt and track the shift.
    """
    print("=" * 60)
    print("TEST 2: Regime change — beta shifts from 1.0 to 2.0")
    print("=" * 60)

    np.random.seed(123)
    n = 600
    switch = 300

    log_b = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.85 * noise[i - 1] + np.random.normal(0, 0.005)

    # First half: beta = 1.0, second half: beta = 2.0
    log_a = np.empty(n)
    log_a[:switch] = 1.0 * log_b[:switch] + 1.0 + noise[:switch]
    log_a[switch:] = 2.0 * log_b[switch:] + 1.0 + noise[switch:]

    beta = kalman_hedge_ratio(log_a, log_b, delta=1e-4)

    beta_first_half = np.mean(beta[200:290])   # settled in first regime
    beta_second_half = np.mean(beta[450:590])  # settled in second regime

    print(f"  First regime beta (expect ~1.0):  {beta_first_half:.4f}")
    print(f"  Second regime beta (expect ~2.0): {beta_second_half:.4f}")
    print(f"  Adaptation detected:              {beta_second_half > beta_first_half + 0.3}")

    assert beta_first_half < 1.5, \
        f"First half should be closer to 1.0, got {beta_first_half:.4f}"
    assert beta_second_half > 1.5, \
        f"Second half should be closer to 2.0, got {beta_second_half:.4f}"

    print("  PASSED\n")


def test_3_kalman_spread_stationarity():
    """
    Test: on data with a DRIFTING beta, Kalman spread should be
    tighter than OLS spread. On constant-beta data, OLS wins because
    it has full hindsight — that's expected, not a bug.
    """
    print("=" * 60)
    print("TEST 3: Kalman vs OLS on drifting beta")
    print("=" * 60)

    np.random.seed(77)
    n = 500

    log_b = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0

    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + np.random.normal(0, 0.005)

    # Beta slowly drifts from 1.3 to 1.6 over the series
    true_beta = np.linspace(1.3, 1.6, n)
    log_a = true_beta * log_b + 1.0 + noise

    # OLS spread (static beta — can't track the drift)
    ols_beta = np.polyfit(log_b, log_a, deg=1)[0]
    ols_spread = log_a - ols_beta * log_b

    # Kalman spread (dynamic beta — should track the drift)
    k_spread, k_beta = kalman_spread(log_a, log_b, delta=1e-4)

    # Skip warmup (first 100 points for Kalman to settle)
    ols_std = np.std(ols_spread[100:])
    kalman_std = np.std(k_spread[100:])

    print(f"  True beta drifts:    1.3 -> 1.6")
    print(f"  OLS beta (fixed):    {ols_beta:.4f}")
    print(f"  Kalman beta (final): {k_beta[-1]:.4f}")
    print(f"  OLS spread std:      {ols_std:.6f}")
    print(f"  Kalman spread std:   {kalman_std:.6f}")
    print(f"  Kalman tighter:      {kalman_std < ols_std}")

    assert kalman_std < ols_std, \
        f"Kalman ({kalman_std:.6f}) should beat OLS ({ols_std:.6f}) on drifting beta"

    print("  PASSED\n")


def test_4_real_pairs():
    """
    Test: run on real discovered pairs if the data exists.
    Compares ADF p-values for OLS vs Kalman spreads.
    """
    print("=" * 60)
    print("TEST 4: Real pairs from discovered_pairs.csv")
    print("=" * 60)

    prices_path = PROJECT_ROOT / "data" / "processed" / "prices_features.csv"
    pairs_path = PROJECT_ROOT / "data" / "processed" / "discovered_pairs.csv"

    if not prices_path.exists() or not pairs_path.exists():
        print("  [SKIP] Data files not found. Run the pipeline first.")
        print(f"    Expected: {prices_path}")
        print(f"    Expected: {pairs_path}")
        return

    print("  Loading data...")
    full_df = pd.read_csv(prices_path, parse_dates=["Date"])
    pairs_df = pd.read_csv(pairs_path)

    # Use the first training window's pairs
    window_label = "2010_2012"
    window_pairs = pairs_df[
        (pairs_df["training_window"] == window_label)
        & (pairs_df["is_eligible"].astype(bool))
    ]

    if window_pairs.empty:
        print(f"  [SKIP] No eligible pairs for window {window_label}")
        return

    # Filter prices to training window
    window_df = full_df[
        (full_df["Date"] >= "2010-01-01")
        & (full_df["Date"] <= "2012-12-31")
    ]
    prices_pivot = window_df.pivot_table(
        index="Date", columns="Ticker", values="Close", aggfunc="last"
    ).sort_index()

    # Run comparison on top 10 pairs
    top_pairs = window_pairs.head(10)
    comparison = kalman_vs_ols_comparison(prices_pivot, top_pairs, delta=1e-4)

    if comparison.empty:
        print("  [SKIP] No pairs could be compared (missing price data)")
        return

    print(f"\n  Compared {len(comparison)} pairs:")
    print(f"  {'Pair':<20} {'OLS ADF p':>10} {'Kalman ADF p':>12} {'Kalman better?':>15}")
    print("  " + "-" * 60)

    for _, row in comparison.iterrows():
        better = row.get("kalman_more_stationary", None)
        marker = "Yes" if better else ("No" if better is False else "N/A")
        print(
            f"  {row['pair']:<20}"
            f" {row['ols_adf_pval']:>10.4f}"
            f" {row['kalman_adf_pval']:>12.4f}"
            f" {marker:>15}"
        )

    n_better = comparison["kalman_more_stationary"].sum()
    n_total = comparison["kalman_more_stationary"].notna().sum()
    print(f"\n  Kalman more stationary: {n_better}/{n_total} pairs")
    print("  (If most pairs show Kalman is better, the feature adds value)")
    print("  DONE\n")


def test_5_delta_sensitivity():
    """
    Test: how sensitive is the Kalman beta to the delta parameter?
    Smaller delta = smoother beta, larger delta = more reactive.
    """
    print("=" * 60)
    print("TEST 5: Delta sensitivity")
    print("=" * 60)

    np.random.seed(99)
    n = 400

    log_b = np.cumsum(np.random.normal(0, 0.01, n)) + 4.0
    noise = np.zeros(n)
    for i in range(1, n):
        noise[i] = 0.9 * noise[i - 1] + np.random.normal(0, 0.005)
    log_a = 1.5 * log_b + 1.0 + noise

    print(f"  {'Delta':<12} {'Beta mean':>10} {'Beta std':>10} {'Interpretation'}")
    print("  " + "-" * 55)

    for delta in [1e-6, 1e-5, 1e-4, 1e-3, 1e-2]:
        beta = kalman_hedge_ratio(log_a, log_b, delta=delta)
        b_mean = np.mean(beta[50:])
        b_std = np.std(beta[50:])
        if b_std < 0.01:
            interp = "Very smooth (slow to adapt)"
        elif b_std < 0.05:
            interp = "Good balance"
        else:
            interp = "Noisy (overreacts)"
        print(f"  {delta:<12.0e} {b_mean:>10.4f} {b_std:>10.4f} {interp}")

    print("\n  Recommended: delta=1e-4 (default). Increase to 1e-3 if pairs are unstable.")
    print("  DONE\n")


if __name__ == "__main__":
    test_1_synthetic_stable_beta()
    test_2_regime_change()
    test_3_kalman_spread_stationarity()
    test_4_real_pairs()
    test_5_delta_sensitivity()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
