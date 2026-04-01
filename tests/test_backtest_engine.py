"""
Synthetic test for backtest_engine.py
======================================
Run from project root:
    python tests/test_backtest_engine.py

Generates two synthetic cointegrated pairs (no real data needed) and
injects them directly into BacktestEngine, bypassing load_data().
Shows the full report() and plot() output so you can verify the engine works.
"""

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import BacktestEngine, BacktestConfig, ZScoreSignal


# ── Synthetic data generation ──────────────────────────────────────────────────

def make_cointegrated_pair(
    n: int,
    beta: float,
    half_life: float,
    seed: int,
    start: str = "2010-01-02",
) -> tuple[pd.Series, pd.Series]:
    """
    Generate two cointegrated price series.
      - Stock B: geometric random walk
      - Stock A: beta * B + mean-reverting spread with given half-life
    """
    rng = np.random.default_rng(seed)
    dates = pd.bdate_range(start=start, periods=n)

    # B: random walk
    log_b = np.cumsum(rng.normal(0, 0.01, n)) + 4.0
    b = np.exp(log_b)

    # spread: OU process with given half-life
    theta = np.log(2) / half_life
    spread = np.zeros(n)
    for i in range(1, n):
        spread[i] = spread[i - 1] * (1 - theta) + rng.normal(0, 0.02)

    # A: beta * B + spread
    log_a = beta * log_b + spread
    a = np.exp(log_a)

    return (
        pd.Series(a, index=dates, name="A"),
        pd.Series(b, index=dates, name="B"),
    )


def build_synthetic_engine(cfg: BacktestConfig) -> BacktestEngine:
    """
    Build a BacktestEngine pre-loaded with two synthetic cointegrated pairs,
    one per training window, without touching any files.
    """
    # Window definitions matching config.py
    windows = [
        ("2010_2012", "2010-01-02", "2012-12-31", "2013-01-02", "2013-12-31"),
        ("2010_2013", "2010-01-02", "2013-12-31", "2014-01-02", "2014-12-31"),
        ("2010_2014", "2010-01-02", "2014-12-31", "2015-01-02", "2015-12-31"),
        ("2010_2015", "2010-01-02", "2015-12-31", "2016-01-02", "2016-12-31"),
        ("2010_2016", "2010-01-02", "2016-12-31", "2017-01-02", "2017-12-31"),
    ]

    all_prices: dict[str, pd.Series] = {}
    pair_rows = []

    for i, (label, train_start, train_end, val_start, val_end) in enumerate(windows):
        n_train = len(pd.bdate_range(train_start, train_end))
        n_total = len(pd.bdate_range(train_start, val_end))

        # Two pairs per window, different betas and half-lives
        for j, (beta, hl, suffix) in enumerate([(1.2, 15, "X"), (0.8, 25, "Y")]):
            a_name = f"A{label[-4:]}_{suffix}"
            b_name = f"B{label[-4:]}_{suffix}"

            a, b = make_cointegrated_pair(n_total, beta, hl, seed=i * 10 + j, start=train_start)
            a.name = a_name
            b.name = b_name

            all_prices[a_name] = a
            all_prices[b_name] = b

            pair_rows.append({
                "pair":            f"{a_name}-{b_name}",
                "training_window": label,
                "is_eligible":     True,
                "score":           0.80 - j * 0.05,
                "initial_beta":    beta,
                "cluster":         0,
                "coint_pval":      0.01,
                "half_life":       hl,
                "hurst":           0.4,
                "mean_crossings":  20,
                "mean_intercept":  0.0,
                "pearson":         0.95,
                "spearman":        0.94,
            })

    # Wide price DataFrame (Date × Ticker)
    wide = pd.DataFrame(all_prices).sort_index()

    # Pairs DataFrame
    pairs_df = pd.DataFrame(pair_rows)
    pairs_df["window_pair_id"] = pairs_df["pair"] + "_" + pairs_df["training_window"]

    engine = BacktestEngine(cfg)
    engine._wide = wide
    engine.pairs = pairs_df

    return engine


# ── Tests ──────────────────────────────────────────────────────────────────────

def test_zscore_signal():
    print("=" * 60)
    print("TEST: ZScoreSignal on synthetic cointegrated pairs")
    print("=" * 60)

    cfg = BacktestConfig(
        entry_z=2.0,
        exit_z=0.5,
        stop_z=4.0,
        n_top_pairs=10,
        initial_capital=1_000_000.0,
        transaction_cost_bps=10.0,
    )

    engine = build_synthetic_engine(cfg)

    print("\n── Expanding-window CV ──")
    results = engine.run()
    engine.report(results)

    assert "__aggregate__" in results, "No aggregate results produced"
    agg_metrics = results["__aggregate__"]["metrics"]
    assert "sharpe" in agg_metrics
    assert "fitness" in agg_metrics
    print("Metrics keys OK\n")


def test_holdout():
    print("=" * 60)
    print("TEST: Holdout test (2017)")
    print("=" * 60)

    cfg = BacktestConfig(n_top_pairs=10, initial_capital=1_000_000.0)
    engine = build_synthetic_engine(cfg)

    results = engine.run_holdout()
    engine.report(results)

    assert "__aggregate__" in results, "No holdout results produced"
    print("Holdout OK\n")


def test_rolling_zscore():
    print("=" * 60)
    print("TEST: Rolling z-score variant")
    print("=" * 60)

    cfg = BacktestConfig(
        use_rolling_zscore=True,
        rolling_lookback=63,
        n_top_pairs=10,
    )
    engine = build_synthetic_engine(cfg)
    results = engine.run()
    engine.report(results)

    assert "__aggregate__" in results
    print("Rolling z-score OK\n")


def test_custom_signal():
    print("=" * 60)
    print("TEST: Custom ZScoreSignal with tighter thresholds")
    print("=" * 60)

    cfg = BacktestConfig(n_top_pairs=10)
    engine = build_synthetic_engine(cfg)

    custom = ZScoreSignal(entry_z=1.5, exit_z=0.3, stop_z=3.0)
    results = engine.run(signal_generator=custom)
    engine.report(results)

    assert "__aggregate__" in results
    print("Custom signal OK\n")


if __name__ == "__main__":
    test_zscore_signal()
    test_holdout()
    test_rolling_zscore()
    test_custom_signal()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
