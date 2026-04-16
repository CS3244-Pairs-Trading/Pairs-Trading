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
import pytest

from src.backtest.backtest_engine import BacktestEngine, BacktestConfig, ZScoreSignal, execute_signals


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
        ("2011_2013", "2011-01-02", "2013-12-31", "2014-01-02", "2014-12-31"),
        ("2012_2014", "2012-01-02", "2014-12-31", "2015-01-02", "2015-12-31"),
        ("2013_2015", "2013-01-02", "2015-12-31", "2016-01-02", "2016-12-31"),
        ("2014_2016", "2014-01-02", "2016-12-31", "2017-01-02", "2017-12-31"),
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
                "stock_a":         a_name,
                "stock_b":         b_name,
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


def test_execute_signals_forced_close_books_cost():
    dates = pd.bdate_range("2017-01-02", periods=3)
    c1 = pd.Series([100.0, 101.0, 102.0], index=dates)
    c2 = pd.Series([100.0, 100.5, 101.0], index=dates)
    signals = pd.Series([0, 1, 1], index=dates)

    pnl, tv, _, _ = execute_signals(
        c1,
        c2,
        signals,
        beta=1.0,
        cfg=BacktestConfig(transaction_cost_bps=10.0),
        allocation=10_000.0,
    )

    close_cost = 10_000.0 * (10.0 / 10_000.0) * 2.0
    assert tv.iloc[-1] > 0.0, "Forced close should contribute turnover on the last day"
    assert pnl.iloc[-1] < 24.0, "Forced close cost should reduce final-day P&L"
    assert pnl.iloc[-1] == pytest.approx(24.629328604501843 - close_cost, rel=1e-6)


class DummyCoverageSignal:
    def __init__(self, tradable_pairs: set[str]) -> None:
        self.tradable_pairs = tradable_pairs

    def has_pair_predictions(self, window_label: str, pair_name: str) -> bool:
        return pair_name in self.tradable_pairs

    def fit(self, c1_train, c2_train, stats):
        self._pair = stats["pair"]

    def predict(self, c1_test, c2_test):
        arr = np.zeros(len(c1_test), dtype=int)
        if len(arr) > 1:
            arr[1:] = 1
        return pd.Series(arr, index=c1_test.index)


def test_engine_allocates_only_across_tradable_pairs():
    dates = pd.bdate_range("2014-01-02", periods=90)
    price_cols = {
        "A1": pd.Series(np.linspace(100.0, 110.0, len(dates)), index=dates),
        "B1": pd.Series(np.linspace(95.0, 100.0, len(dates)), index=dates),
        "A2": pd.Series(np.linspace(80.0, 84.0, len(dates)), index=dates),
        "B2": pd.Series(np.linspace(75.0, 79.0, len(dates)), index=dates),
    }

    engine = BacktestEngine(BacktestConfig(n_top_pairs=2, initial_capital=100_000.0))
    engine._wide = pd.DataFrame(price_cols)
    engine.pairs = pd.DataFrame(
        [
            {
                "pair": "A1-B1",
                "stock_a": "A1",
                "stock_b": "B1",
                "training_window": "2014_2016",
                "is_eligible": True,
                "score": 0.9,
                "initial_beta": 1.0,
            },
            {
                "pair": "A2-B2",
                "stock_a": "A2",
                "stock_b": "B2",
                "training_window": "2014_2016",
                "is_eligible": True,
                "score": 0.8,
                "initial_beta": 1.0,
            },
        ]
    )

    res = engine._run_window(
        "2014_2016",
        train_end="2014-04-30",
        test_start="2014-05-01",
        test_end="2014-05-07",
        signal_generator=DummyCoverageSignal({"A1-B1"}),
    )

    assert res is not None
    assert res["n_pairs_selected"] == 2
    assert res["n_pairs_tradable"] == 1


if __name__ == "__main__":
    test_zscore_signal()
    test_holdout()
    test_rolling_zscore()
    test_custom_signal()

    print("=" * 60)
    print("ALL TESTS COMPLETE")
    print("=" * 60)
