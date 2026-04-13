#!/usr/bin/env python3
"""
Pairs Trading Backtest Engine
==============================
Evaluates pairs discovered by rank_pairs.py using a pluggable signal model
across expanding validation windows.

Architecture
-------------
Signal generation and trade execution are intentionally separated so that
different models (OU/Z-score, ARMA, XGBoost, LSTM, …) can be compared
under identical execution conditions.

  SignalGenerator  (Protocol)
    ├── fit(c1_train, c2_train, stats)  → train on historical data
    └── predict(c1_test, c2_test)       → pd.Series of {-1, 0, +1}

  execute_signals(c1, c2, signals, beta, cfg, allocation)
    └── P&L, transaction costs, turnover, bookkeeping
        *** identical for every model — the only fair comparison ***

Built-in signal generators
----------------------------
  ZScoreSignal   – classic z-score mean-reversion  (default)

Adding a new model
-------------------
  class MyModel:
      def fit(self, c1_train, c2_train, stats): ...
      def predict(self, c1_test, c2_test) -> pd.Series: ...
          # return pd.Series with values in {-1, 0, +1}

  engine.run(signal_generator=MyModel())

Metrics
-----------------------------------------
  Sharpe   – annualised risk-adjusted return
  Turnover – annualised fraction of book traded
  Fitness  – Sharpe × √( |Returns| / max(Turnover, 12.5%) )
  Margin   – ( Returns / Turnover ) × 20  [per-mille, ‰]
  Returns  – annualised
  Drawdown – maximum peak-to-trough

Quick-start
-----------
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig, ZScoreSignal

    engine = BacktestEngine(BacktestConfig(entry_z=2.0, exit_z=0.5))
    engine.load_data(
        "data/processed/prices_features.csv",
        "data/processed/discovered_pairs.csv",
    )

    # default: z-score signal built from BacktestConfig parameters
    results = engine.run()
    results = engine.run_holdout()

    # custom signal model
    results = engine.run(signal_generator=ZScoreSignal(entry_z=1.5, exit_z=0.3))

    engine.report(results)
    engine.plot(results)
    engine.save(results)

CLI
----
    python -m src.backtest.backtest_engine --help
    python -m src.backtest.backtest_engine
    python -m src.backtest.backtest_engine --holdout --save
    python -m src.backtest.backtest_engine --entry_z 1.5 --exit_z 0.3 --n_pairs 30
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Optional, Protocol, runtime_checkable

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, DEFAULT_BACKTEST_PARAMS

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# CONSTANTS  (sourced from config.py — change values there)
# ──────────────────────────────────────────────────────────
TRADING_DAYS     = DEFAULT_BACKTEST_PARAMS.trading_days
RISK_FREE_RATE   = DEFAULT_BACKTEST_PARAMS.risk_free_rate
FITNESS_TV_FLOOR = DEFAULT_BACKTEST_PARAMS.fitness_tv_floor


# ──────────────────────────────────────────────────────────
# CONFIGURATION
# ──────────────────────────────────────────────────────────
@dataclass
class BacktestConfig:
    """
    Execution and capital parameters for the backtest.

    The z-score fields (entry_z, exit_z, stop_z, use_rolling_zscore,
    rolling_lookback) are convenience defaults used when no explicit
    signal_generator is passed to run() / run_holdout(). They have no
    effect when a custom SignalGenerator is supplied.
    """

    # ── z-score signal defaults (used by ZScoreSignal when auto-created) ──
    entry_z:            float = DEFAULT_BACKTEST_PARAMS.entry_z
    exit_z:             float = DEFAULT_BACKTEST_PARAMS.exit_z
    stop_z:             float = DEFAULT_BACKTEST_PARAMS.stop_z
    use_rolling_zscore: bool  = DEFAULT_BACKTEST_PARAMS.use_rolling_zscore
    rolling_lookback:   int   = DEFAULT_BACKTEST_PARAMS.rolling_lookback

    # ── capital & sizing ────────────────────────────────────────────────
    initial_capital: float = DEFAULT_BACKTEST_PARAMS.initial_capital
    n_top_pairs:     int   = DEFAULT_BACKTEST_PARAMS.n_top_pairs

    # ── cost model ──────────────────────────────────────────────────────
    transaction_cost_bps: float = DEFAULT_BACKTEST_PARAMS.transaction_cost_bps

    # ── pair selection ───────────────────────────────────────────────────
    min_score: float = DEFAULT_BACKTEST_PARAMS.min_pair_score

    # ── output ───────────────────────────────────────────────────────────
    output_dir: str = DEFAULT_BACKTEST_PARAMS.output_dir


# ──────────────────────────────────────────────────────────
# PERFORMANCE METRICS
# ──────────────────────────────────────────────────────────
def compute_metrics(
    daily_returns:  pd.Series,
    n_trades:       int,
    daily_turnover: pd.Series,
    risk_free:      float = RISK_FREE_RATE,
) -> dict:
    """
    Compute performance metrics

    Parameters
    ----------
    daily_returns  : daily P&L / initial_capital  (decimal, signed)
    n_trades       : total trades (entries + exits) over the period
    daily_turnover : fraction of book traded each day (decimal, one-sided)
    risk_free      : annualised risk-free rate in decimal

    Returns
    -------
    dict with keys:
        total_return, annualized_return, sharpe, max_drawdown,
        volatility, n_trades, turnover, margin_permille, fitness
    """
    n = len(daily_returns)
    if n == 0:
        return _empty_metrics(n_trades)

    n_years = n / TRADING_DAYS

    # returns
    total_return = float((1 + daily_returns).prod() - 1)
    ann_return   = float((1 + total_return) ** (1 / n_years) - 1) if n_years > 0 else 0.0

    # risk
    volatility = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe     = (ann_return - risk_free) / volatility if volatility > 0 else 0.0

    # drawdown
    cum    = (1 + daily_returns).cumprod()
    max_dd = float((cum / cum.cummax() - 1).min())

    # turnover (annualised)
    ann_tv = float(daily_turnover.mean() * TRADING_DAYS) if len(daily_turnover) > 0 else 0.0

    # Margin [‰] = (ann_return / ann_turnover) × margin_multiplier
    margin = (ann_return / ann_tv * DEFAULT_BACKTEST_PARAMS.margin_multiplier) if ann_tv > 0 else 0.0

    # Fitness = Sharpe × √( |ann_return| / max(ann_turnover, 12.5%) )
    fitness = (
        sharpe * np.sqrt(abs(ann_return) / max(ann_tv, FITNESS_TV_FLOOR))
        if volatility > 0 and ann_tv > 0 else 0.0
    )

    return {
        "total_return":      total_return,
        "annualized_return": ann_return,
        "sharpe":            sharpe,
        "max_drawdown":      max_dd,
        "volatility":        volatility,
        "n_trades":          n_trades,
        "turnover":          ann_tv,
        "margin_permille":   margin,
        "fitness":           fitness,
    }


def _empty_metrics(n_trades: int = 0) -> dict:
    return {
        "total_return": 0.0, "annualized_return": 0.0, "sharpe": 0.0,
        "max_drawdown": 0.0, "volatility":        0.0, "n_trades": n_trades,
        "turnover":     0.0, "margin_permille":   0.0, "fitness": 0.0,
    }


# ──────────────────────────────────────────────────────────
# SHARED UTILITIES
# ──────────────────────────────────────────────────────────
def _spread_stats(c1_train: pd.Series, c2_train: pd.Series, beta: float) -> dict:
    """Compute spread mean / std on training data only (no look-ahead)."""
    spread = c1_train - beta * c2_train
    return {
        "mean": float(spread.mean()),
        "std":  float(spread.std()),
        "beta": beta,
    }


def _zscore(
    spread:      pd.Series,
    stats:       dict,
    use_rolling: bool,
    lookback:    int,
) -> pd.Series:
    """Return z-score series using fixed training stats or a rolling window."""
    if use_rolling:
        min_p = max(1, lookback // 2)
        mu    = spread.rolling(lookback, min_periods=min_p).mean()
        sig   = spread.rolling(lookback, min_periods=min_p).std().clip(lower=1e-8)
    else:
        mu  = stats["mean"]
        sig = max(stats["std"], 1e-8)
    return (spread - mu) / sig


# ──────────────────────────────────────────────────────────
# SIGNAL GENERATOR PROTOCOL
# ──────────────────────────────────────────────────────────
@runtime_checkable
class SignalGenerator(Protocol):
    """
    Interface that every signal model must implement.

    The engine calls fit() once on training data, then predict() on the
    test period. The returned series drives execute_signals(), which is
    identical for all models — ensuring fair comparison.
    """

    def fit(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
        stats:    dict,
    ) -> None:
        """
        Train the model on historical data.

        Parameters
        ----------
        c1_train, c2_train : training-period Close prices (aligned index)
        stats              : dict from _spread_stats() — keys: mean, std, beta
                             Available to all models for spread normalisation.
        """
        ...

    def predict(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> pd.Series:
        """
        Generate position signals for the test period.

        Parameters
        ----------
        c1_test, c2_test : test-period Close prices (aligned index)

        Returns
        -------
        pd.Series  same index as c1_test, integer values in {-1, 0, +1}
            +1  long  spread  (long c1, short c2)
            -1  short spread  (short c1, long c2)
             0  flat  (no position)
        """
        ...


# ──────────────────────────────────────────────────────────
# BUILT-IN SIGNAL GENERATORS
# ──────────────────────────────────────────────────────────
class ZScoreSignal:
    """
    Classic z-score mean-reversion signal (default model).

    Mirrors the original simulate_pair() signal logic exactly:
      - Enter long  when z < -entry_z
      - Enter short when z > +entry_z
      - Exit         when |z| < exit_z
      - Stop-loss    when |z| > stop_z  (disabled if stop_z=0)
      - Reverse      when z crosses to the opposite entry threshold
    """

    def __init__(
        self,
        entry_z:          float = 2.0,
        exit_z:           float = 0.5,
        stop_z:           float = 4.0,
        use_rolling:      bool  = False,
        rolling_lookback: int   = 63,
    ) -> None:
        self.entry_z          = entry_z
        self.exit_z           = exit_z
        self.stop_z           = stop_z
        self.use_rolling      = use_rolling
        self.rolling_lookback = rolling_lookback
        self._stats: dict     = {}

    def fit(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
        stats:    dict,
    ) -> None:
        """Store training-period spread statistics for z-score computation."""
        self._stats = stats

    def predict(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> pd.Series:
        """
        Compute z-score on the test period and apply threshold rules.

        Day 0 is always flat (matching original simulate_pair behaviour
        which starts the signal loop at i=1).
        """
        beta   = self._stats["beta"]
        spread = c1_test - beta * c2_test
        z      = _zscore(spread, self._stats, self.use_rolling, self.rolling_lookback)

        n       = len(c1_test)
        sig_arr = np.zeros(n, dtype=int)
        pos     = 0

        # start from i=1 (day 0 always flat, consistent with execute_signals)
        for i in range(1, n):
            zi = float(z.iloc[i])
            if np.isnan(zi):
                sig_arr[i] = pos
                continue

            if pos == 0:
                if zi <= -self.entry_z:
                    pos = 1
                elif zi >= self.entry_z:
                    pos = -1
            else:
                if abs(zi) <= self.exit_z:
                    pos = 0
                elif self.stop_z > 0 and abs(zi) >= self.stop_z:
                    pos = 0
                elif pos == 1 and zi >= self.entry_z:
                    pos = -1    # reverse: long → short
                elif pos == -1 and zi <= -self.entry_z:
                    pos = 1     # reverse: short → long

            sig_arr[i] = pos

        return pd.Series(sig_arr, index=c1_test.index)


class PredictionSignal:
    """
    Adaptive ML-prediction signal with per-pair quantile-calibrated thresholds.

    Consumes pre-computed model predictions from
    data/processed/predictions/<model_name>/<window_label>/predictions.csv
    (columns: Date, pair, predicted_change or predicted_spread_change).

    Rather than hardcoding z-thresholds, entry / exit / stop levels are
    learned per pair from the empirical distribution of that pair's
    predicted_change values over a causal warmup window at the start of
    the test period. This mirrors QuantileZScoreSignal in spirit — the
    same rule works for pairs whose predicted_change is tiny, noisy, or
    highly skewed, because the thresholds are self-scaled.

    Direction convention matches the engine: +1 long spread when the
    model expects the spread to rise, -1 short when it expects it to
    fall. Warmup days are held flat (no look-ahead).
    """

    def __init__(
        self,
        predictions_root: str | Path,
        entry_quantile:   float = DEFAULT_BACKTEST_PARAMS.entry_quantile,
        exit_quantile:    float = DEFAULT_BACKTEST_PARAMS.exit_quantile,
        stop_quantile:    float = DEFAULT_BACKTEST_PARAMS.stop_quantile,
        warmup_days:      int   = 60,
        min_obs:          int   = DEFAULT_BACKTEST_PARAMS.quantile_min_obs,
    ) -> None:
        self.predictions_root = Path(predictions_root)
        self.entry_quantile   = entry_quantile
        self.exit_quantile    = exit_quantile
        self.stop_quantile    = stop_quantile
        self.warmup_days      = warmup_days
        self.min_obs          = min_obs

        self._window_cache: dict[str, Optional[pd.DataFrame]] = {}
        self._current_preds: Optional[pd.Series]              = None
        self.n_missing_pairs  = 0
        self.n_matched_pairs  = 0

    def _load_window(self, window_label: str) -> Optional[pd.DataFrame]:
        if window_label in self._window_cache:
            return self._window_cache[window_label]

        path = self.predictions_root / window_label / "predictions.csv"
        if not path.exists():
            self._window_cache[window_label] = None
            return None

        df = pd.read_csv(path, parse_dates=["Date"])
        if "predicted_change" not in df.columns:
            if "predicted_spread_change" in df.columns:
                df = df.rename(columns={"predicted_spread_change": "predicted_change"})
            else:
                raise ValueError(
                    f"{path} missing predicted_change / predicted_spread_change column"
                )

        self._window_cache[window_label] = df
        return df

    def fit(
        self,
        c1_train: pd.Series,
        c2_train: pd.Series,
        stats:    dict,
    ) -> None:
        pair   = stats.get("pair")
        window = stats.get("window")
        if pair is None or window is None:
            raise ValueError(
                "PredictionSignal requires 'pair' and 'window' in the stats dict "
                "passed to fit() — make sure the engine injects them."
            )

        df = self._load_window(window)
        if df is None:
            self._current_preds = None
            return

        sub = df.loc[df["pair"] == pair, ["Date", "predicted_change"]]
        if sub.empty:
            self._current_preds = None
            self.n_missing_pairs += 1
            return

        self._current_preds = (
            sub.dropna(subset=["predicted_change"])
               .set_index("Date")["predicted_change"]
               .sort_index()
        )
        self.n_matched_pairs += 1

    def _calibrate(self, warmup: pd.Series) -> Optional[dict[str, float]]:
        """
        Derive signed entry/exit/stop thresholds from a warmup distribution.
        Returns None if the distribution is unusable.
        """
        warmup = warmup.dropna()
        if len(warmup) < self.min_obs:
            return None

        upper_entry = float(warmup.quantile(1.0 - self.entry_quantile))
        lower_entry = float(warmup.quantile(self.entry_quantile))
        upper_exit  = float(warmup.quantile(1.0 - self.exit_quantile))
        lower_exit  = float(warmup.quantile(self.exit_quantile))

        # Sanity: entry bands must sit outside the exit band
        if upper_entry <= upper_exit or lower_entry >= lower_exit:
            return None

        if self.stop_quantile > 0:
            upper_stop = float(warmup.quantile(1.0 - self.stop_quantile))
            lower_stop = float(warmup.quantile(self.stop_quantile))
            # Sanity: stops must sit outside the entry band
            if upper_stop <= upper_entry or lower_stop >= lower_entry:
                upper_stop = lower_stop = None  # disable stops
        else:
            upper_stop = lower_stop = None

        return {
            "upper_entry": upper_entry,
            "lower_entry": lower_entry,
            "upper_exit":  upper_exit,
            "lower_exit":  lower_exit,
            "upper_stop":  upper_stop,
            "lower_stop":  lower_stop,
        }

    def predict(
        self,
        c1_test: pd.Series,
        c2_test: pd.Series,
    ) -> pd.Series:
        n   = len(c1_test)
        arr = np.zeros(n, dtype=int)

        preds = self._current_preds
        if preds is None or preds.empty:
            return pd.Series(arr, index=c1_test.index)

        change = preds.reindex(c1_test.index)

        # Causal warmup: fit thresholds on the first `warmup_days` predictions.
        warmup_end = min(self.warmup_days, n)
        thresholds = self._calibrate(change.iloc[:warmup_end])
        if thresholds is None:
            return pd.Series(arr, index=c1_test.index)

        ue = thresholds["upper_entry"]
        le = thresholds["lower_entry"]
        ux = thresholds["upper_exit"]
        lx = thresholds["lower_exit"]
        us = thresholds["upper_stop"]
        ls = thresholds["lower_stop"]

        pos = 0
        for i in range(warmup_end, n):
            val_raw = change.iloc[i]
            if pd.isna(val_raw):
                arr[i] = pos
                continue
            val = float(val_raw)

            if pos == 0:
                if val >= ue:
                    pos = 1       # strong upward prediction → long spread
                elif val <= le:
                    pos = -1      # strong downward prediction → short spread
            else:
                if lx <= val <= ux:
                    pos = 0       # prediction weakens → exit
                elif us is not None and (val >= us or val <= ls):
                    pos = 0       # extreme prediction → likely noise, bail
                elif pos == 1 and val <= le:
                    pos = -1      # reverse long → short
                elif pos == -1 and val >= ue:
                    pos = 1       # reverse short → long

            arr[i] = pos

        return pd.Series(arr, index=c1_test.index)


# ──────────────────────────────────────────────────────────
# EXECUTION ENGINE  (model-agnostic)
# ──────────────────────────────────────────────────────────
def execute_signals(
    c1:         pd.Series,
    c2:         pd.Series,
    signals:    pd.Series,
    beta:       float,
    cfg:        BacktestConfig,
    allocation: float,
) -> tuple[pd.Series, pd.Series, dict, dict]:
    """
    Execute a pre-computed signal series — completely model-agnostic.

    This is the single execution path used for every signal model.
    Keeping it identical across models is what makes metric comparisons fair.

    Position sizing (beta-adjusted, dollar-neutral):
        long  leg  →  k1 = allocation / (1 + |beta|)   dollars in c1
        short leg  →  k2 = k1 × |beta|                 dollars in c2
        total book                                      = allocation  (always)

    Parameters
    ----------
    c1, c2    : test-period Close prices (aligned index)
    signals   : pd.Series {-1, 0, +1} produced by any SignalGenerator.predict()
    beta      : hedge ratio from training data
    cfg       : BacktestConfig  (used for transaction_cost_bps only)
    allocation: dollar allocation for this pair

    Returns
    -------
    daily_pnl   : pd.Series  daily P&L in dollars
    daily_tv    : pd.Series  dollars traded each day (for turnover)
    n_long_yr   : dict {year → long-side trade count}
    n_short_yr  : dict {year → short-side trade count}
    """
    denom   = 1.0 + abs(beta)
    k1      = allocation / denom
    k2      = allocation * abs(beta) / denom
    tc_rate = cfg.transaction_cost_bps / 10_000.0

    ret1 = c1.pct_change().fillna(0.0)
    ret2 = c2.pct_change().fillna(0.0)

    n       = len(c1)
    pnl_arr = np.zeros(n)
    tv_arr  = np.zeros(n)

    n_long_yr:  dict[int, int] = {}
    n_short_yr: dict[int, int] = {}

    pos = 0   # position at start of day (always flat before test period)

    for i in range(1, n):
        new_pos = int(signals.iloc[i])

        # ── P&L: earned from the position held at the start of day i ────
        if pos != 0:
            pnl_arr[i] = pos * (
                k1 * float(ret1.iloc[i]) - k2 * float(ret2.iloc[i])
            )

        # ── transaction costs when position changes ──────────────────────
        if new_pos != pos:
            delta          = abs(new_pos - pos)   # 1 = open/close, 2 = reverse
            dollars_traded = delta * allocation
            cost           = dollars_traded * tc_rate * 2.0   # 2 legs
            pnl_arr[i]    -= cost
            tv_arr[i]      = dollars_traded

            yr = int(c1.index[i].year)
            if new_pos == 1  or (new_pos == 0 and pos == -1):
                n_long_yr[yr]  = n_long_yr.get(yr,  0) + 1
            if new_pos == -1 or (new_pos == 0 and pos == 1):
                n_short_yr[yr] = n_short_yr.get(yr, 0) + 1

        pos = new_pos

    # ── force-close any open position at period end ──────────────────────
    if pos != 0:
        yr = int(c1.index[-1].year)
        if pos == 1:
            n_short_yr[yr] = n_short_yr.get(yr, 0) + 1
        else:
            n_long_yr[yr]  = n_long_yr.get(yr,  0) + 1

    idx = c1.index
    return (
        pd.Series(pnl_arr, index=idx, name="pnl"),
        pd.Series(tv_arr,  index=idx, name="tv"),
        n_long_yr,
        n_short_yr,
    )


# ──────────────────────────────────────────────────────────
# BACKTEST ENGINE
# ──────────────────────────────────────────────────────────
class BacktestEngine:
    """
    End-to-end backtesting for the pairs trading strategy.

    The engine consumes two artefacts produced by the upstream pipeline:
      * prices_features.csv  – cleaned daily OHLCV + SimpleReturn + LogPrice
      * discovered_pairs.csv – ranked pairs with training-window metadata

    For each training window it selects the top-N eligible pairs by score,
    fits the signal generator on training data (no look-ahead), and runs
    execute_signals() on the out-of-sample validation period.

    See module docstring for usage examples.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.cfg   = config or BacktestConfig()
        self._wide: Optional[pd.DataFrame] = None   # wide Close  (Date × Ticker)
        self.pairs: Optional[pd.DataFrame] = None   # discovered_pairs.csv

    # ── data loading ──────────────────────────────────────────────────────
    def load_data(
        self,
        prices_path: str | Path = DEFAULT_CONFIG.engineered_features_path,
        pairs_path:  str | Path = DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv",
    ) -> "BacktestEngine":
        """
        Load price data and discovered pairs into the engine.

        Parameters
        ----------
        prices_path : path to prices_features.csv  (long format;
                      required columns: Date, Ticker, Close)
        pairs_path  : path to discovered_pairs.csv  (output of rank_pairs.py)

        Returns self for method chaining.
        """
        prices_path = Path(prices_path)
        pairs_path  = Path(pairs_path)

        print(f"[BacktestEngine] Loading prices  → {prices_path}")
        raw = pd.read_csv(prices_path, parse_dates=["Date"])
        raw.sort_values(["Date", "Ticker"], inplace=True)

        self._wide = (
            raw.pivot_table(index="Date", columns="Ticker",
                            values="Close", aggfunc="last")
               .sort_index()
        )
        print(
            f"  → {self._wide.shape[1]:,} tickers | "
            f"{self._wide.index.min().date()} – {self._wide.index.max().date()}"
        )

        print(f"[BacktestEngine] Loading pairs   → {pairs_path}")
        self.pairs = pd.read_csv(pairs_path)
        n_eligible = int(self.pairs["is_eligible"].sum())
        print(f"  → {len(self.pairs):,} total pairs | {n_eligible:,} eligible\n")

        return self

    # ── single-window backtest ─────────────────────────────────────────────
    def _run_window(
        self,
        window_label:     str,
        train_end:        str,
        test_start:       str,
        test_end:         str,
        signal_generator: SignalGenerator,
    ) -> Optional[dict]:
        """
        Backtest one train/test split with the given signal generator.

        The signal generator is fit independently on each pair's training data,
        then predict() is called on the test period. execute_signals() handles
        the rest identically regardless of model type.

        Returns None when no eligible pairs or price data are found.
        """
        cfg = self.cfg

        # ── 1. Select top pairs ──────────────────────────────────────────
        mask = (
            (self.pairs["training_window"] == window_label)
            & self.pairs["is_eligible"].astype(bool)
            & (self.pairs["score"] >= cfg.min_score)
        )
        window_pairs = (
            self.pairs[mask]
            .sort_values("score", ascending=False)
            .head(cfg.n_top_pairs)
        )

        if window_pairs.empty:
            print(f"  [SKIP] {window_label}: no eligible pairs found")
            return None

        # ── 2. Slice price matrices ──────────────────────────────────────
        train_close = self._wide.loc[:train_end]
        test_close  = self._wide.loc[test_start:test_end]

        if test_close.empty:
            print(f"  [SKIP] {window_label}: no price data for {test_start}–{test_end}")
            return None

        n_pairs    = len(window_pairs)
        allocation = cfg.initial_capital / max(n_pairs, 1)

        print(
            f"[{window_label}]  pairs={n_pairs:3d}  |  "
            f"test {test_start} → {test_end}  ({len(test_close)} days)  |  "
            f"model={type(signal_generator).__name__}"
        )

        # ── 3. Fit signal model + execute for each pair ──────────────────
        all_pnl:    dict[str, pd.Series] = {}
        all_tv:     dict[str, pd.Series] = {}
        n_long_yr:  dict[int, int]       = {}
        n_short_yr: dict[int, int]       = {}

        for _, row in window_pairs.iterrows():
            pair_name = row["pair"]
            s1        = row["stock_a"]
            s2        = row["stock_b"]
            beta      = float(row["initial_beta"])

            if s1 not in self._wide.columns or s2 not in self._wide.columns:
                continue

            # align training data
            tr1    = train_close[s1].dropna()
            tr2    = train_close[s2].dropna()
            tr_idx = tr1.index.intersection(tr2.index)
            if len(tr_idx) < 63:      # need ≥ 3 months to estimate stats
                continue
            s_stats = _spread_stats(tr1.loc[tr_idx], tr2.loc[tr_idx], beta)
            s_stats["pair"]   = pair_name
            s_stats["window"] = window_label

            # align test data
            te1    = test_close[s1].dropna()
            te2    = test_close[s2].dropna()
            te_idx = te1.index.intersection(te2.index)
            if len(te_idx) < 5:
                continue

            te1_a = te1.loc[te_idx]
            te2_a = te2.loc[te_idx]

            # fit on training data, predict on test data
            signal_generator.fit(tr1.loc[tr_idx], tr2.loc[tr_idx], s_stats)
            signals = signal_generator.predict(te1_a, te2_a)

            # execute — identical path for every model
            pnl, tv, lng, sht = execute_signals(
                te1_a, te2_a, signals, beta, cfg, allocation
            )

            all_pnl[pair_name] = pnl
            all_tv[pair_name]  = tv

            for yr, cnt in lng.items():
                n_long_yr[yr]  = n_long_yr.get(yr,  0) + cnt
            for yr, cnt in sht.items():
                n_short_yr[yr] = n_short_yr.get(yr, 0) + cnt

        if not all_pnl:
            print(f"  [SKIP] {window_label}: no pairs produced P&L")
            return None

        # ── 4. Aggregate to portfolio ────────────────────────────────────
        pnl_df   = pd.DataFrame(all_pnl).fillna(0.0)
        tv_df    = pd.DataFrame(all_tv).fillna(0.0)
        port_pnl = pnl_df.sum(axis=1)
        port_tv  = tv_df.sum(axis=1)

        daily_ret = port_pnl / cfg.initial_capital
        daily_tv  = port_tv  / cfg.initial_capital

        n_trades = sum(n_long_yr.values()) + sum(n_short_yr.values())
        metrics  = compute_metrics(daily_ret, n_trades, daily_tv)

        return {
            "window":         window_label,
            "test_start":     test_start,
            "test_end":       test_end,
            "n_pairs":        n_pairs,
            "model":          type(signal_generator).__name__,
            "daily_returns":  daily_ret,
            "daily_turnover": daily_tv,
            "daily_pnl":      port_pnl,
            "n_long_yr":      n_long_yr,
            "n_short_yr":     n_short_yr,
            "n_trades_total": n_trades,
            "metrics":        metrics,
        }

    # ── expanding-window cross-validation ─────────────────────────────────
    def run(
        self,
        windows:          Optional[list[str]]       = None,
        signal_generator: Optional[SignalGenerator] = None,
    ) -> dict:
        """
        Run expanding-window out-of-sample backtest.

        Each window's pairs are discovered only on training data and
        evaluated on the following out-of-sample validation year → no leakage.

        Parameters
        ----------
        windows          : window labels to run (default: all 4 CV folds)
        signal_generator : model to use for signal generation.
                           Defaults to ZScoreSignal built from BacktestConfig
                           parameters when None.

        Returns
        -------
        dict keyed by window label + "__aggregate__"
        """
        if self.pairs is None or self._wide is None:
            raise RuntimeError("Call load_data() before run().")

        generator = signal_generator or self._default_signal_generator()

        cfg     = DEFAULT_CONFIG
        win_map: dict[str, tuple[str, str, str]] = {
            fold.label: (fold.train.end, fold.val.start, fold.val.end)
            for fold in cfg.expanding_folds
        }
        win_map[cfg.holdout_split.label] = (
            cfg.holdout_split.train.end,
            cfg.holdout_split.test.start,
            cfg.holdout_split.test.end,
        )

        if windows is None:
            windows = [f.label for f in cfg.expanding_folds]

        results: dict = {}
        for w in windows:
            if w not in win_map:
                print(f"[WARN] Unknown window '{w}' – skipping")
                continue
            train_end, test_start, test_end = win_map[w]
            res = self._run_window(w, train_end, test_start, test_end, generator)
            if res:
                results[w] = res

        if results:
            results["__aggregate__"] = self._aggregate(results)

        return results

    # ── final holdout test ─────────────────────────────────────────────────
    def run_holdout(
        self,
        signal_generator: Optional[SignalGenerator] = None,
    ) -> dict:
        """
        Run the final holdout test (train 2010–2016, test 2017).

        Uses pairs discovered under window label "2010_2016".
        Run this only once — after all tuning is complete.

        Parameters
        ----------
        signal_generator : model to use; defaults to ZScoreSignal from config.
        """
        if self.pairs is None or self._wide is None:
            raise RuntimeError("Call load_data() before run_holdout().")

        generator = signal_generator or self._default_signal_generator()
        hs  = DEFAULT_CONFIG.holdout_split
        res = self._run_window(hs.label, hs.train.end, hs.test.start, hs.test.end, generator)

        if not res:
            return {}
        return {hs.label: res, "__aggregate__": res}

    # ── helper: build default ZScoreSignal from config ────────────────────
    def _default_signal_generator(self) -> ZScoreSignal:
        """Return a ZScoreSignal initialised from BacktestConfig parameters."""
        cfg = self.cfg
        return ZScoreSignal(
            entry_z          = cfg.entry_z,
            exit_z           = cfg.exit_z,
            stop_z           = cfg.stop_z,
            use_rolling      = cfg.use_rolling_zscore,
            rolling_lookback = cfg.rolling_lookback,
        )

    # ── internal: aggregate helper ─────────────────────────────────────────
    @staticmethod
    def _aggregate(results: dict) -> dict:
        """Stitch daily series across windows and compute aggregate metrics."""
        window_vals = [v for k, v in results.items() if k != "__aggregate__"]

        all_ret = pd.concat(
            [v["daily_returns"]  for v in window_vals if "daily_returns"  in v]
        ).sort_index()
        all_tv = pd.concat(
            [v["daily_turnover"] for v in window_vals if "daily_turnover" in v]
        ).sort_index()

        n_long  = sum(sum(v.get("n_long_yr",  {}).values()) for v in window_vals)
        n_short = sum(sum(v.get("n_short_yr", {}).values()) for v in window_vals)

        return {
            "daily_returns":  all_ret,
            "daily_turnover": all_tv,
            "n_long_total":   n_long,
            "n_short_total":  n_short,
            "n_trades_total": n_long + n_short,
            "metrics":        compute_metrics(all_ret, n_long + n_short, all_tv),
        }

    # ── reporting ──────────────────────────────────────────────────────────
    def report(self, results: dict, by_year: bool = True) -> None:
        agg = results.get("__aggregate__")
        if agg is None:
            print("[report] No results to display.")
            return

        m = agg["metrics"]

        print("\n" + "═" * 72)
        print("  BACKTEST SUMMARY")
        print("═" * 72)
        print(f"  {'Total Return':<28}  {m['total_return']:>12.2%}")
        print(f"  {'Annualised Return':<28}  {m['annualized_return']:>12.2%}")
        print(f"  {'Sharpe Ratio':<28}  {m['sharpe']:>12.2f}")
        print(f"  {'Max Drawdown':<28}  {m['max_drawdown']:>12.2%}")
        print(f"  {'Volatility (ann.)':<28}  {m['volatility']:>12.2%}")
        print(f"  {'Number of Trades':<28}  {m['n_trades']:>12,}")
        print(f"  {'Turnover (ann.)':<28}  {m['turnover']:>12.2%}")
        print(f"  {'Margin':<28}  {m['margin_permille']:>10.2f} ‰")
        print(f"  {'Fitness':<28}  {m['fitness']:>12.2f}")
        print("═" * 72)

        if not by_year or "daily_returns" not in agg:
            return

        dr = agg["daily_returns"]
        dt = agg["daily_turnover"]

        lng_yr: dict[int, int] = {}
        sht_yr: dict[int, int] = {}
        for k, v in results.items():
            if k == "__aggregate__":
                continue
            for yr, cnt in v.get("n_long_yr",  {}).items():
                lng_yr[yr] = lng_yr.get(yr, 0) + cnt
            for yr, cnt in v.get("n_short_yr", {}).items():
                sht_yr[yr] = sht_yr.get(yr, 0) + cnt

        print(
            f"\n  {'Year':<6} {'Sharpe':>7} {'Turnover':>10} {'Fitness':>8}"
            f" {'Returns':>9} {'Drawdown':>10} {'Margin':>10}"
            f" {'Long':>6} {'Short':>6}"
        )
        print("  " + "─" * 78)

        for year, grp in dr.groupby(dr.index.year):
            tv_g = dt.reindex(grp.index).fillna(0.0)
            lc   = lng_yr.get(year, 0)
            sc   = sht_yr.get(year, 0)
            ym   = compute_metrics(grp, lc + sc, tv_g)
            print(
                f"  {year:<6}"
                f" {ym['sharpe']:>7.2f}"
                f" {ym['turnover']:>9.2%}"
                f" {ym['fitness']:>8.2f}"
                f" {ym['annualized_return']:>8.2%}"
                f" {ym['max_drawdown']:>9.2%}"
                f" {ym['margin_permille']:>8.2f}‰"
                f" {lc:>6,}"
                f" {sc:>6,}"
            )
        print("  " + "═" * 78 + "\n")

    # ── plotting ───────────────────────────────────────────────────────────
    def plot(self, results: dict, save: bool = True) -> None:
        """Plot cumulative-return curve and drawdown. Saves to output_dir."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("[plot] matplotlib not installed – skipping.")
            return

        agg = results.get("__aggregate__", {})
        if "daily_returns" not in agg:
            print("[plot] No daily returns found in results.")
            return

        dr  = agg["daily_returns"].dropna()
        cum = (1 + dr).cumprod()
        dd  = cum / cum.cummax() - 1

        fig, (ax1, ax2) = plt.subplots(
            2, 1, figsize=(12, 7), sharex=True,
            gridspec_kw={"height_ratios": [3, 1]},
        )
        ax1.plot(cum.index, (cum - 1) * 100, color="#1f77b4", lw=1.5, label="Strategy")
        ax1.axhline(0, color="black", lw=0.7, ls="--")
        ax1.set_ylabel("Cumulative Return (%)")
        ax1.set_title("Pairs Trading – Out-of-Sample Performance")
        ax1.legend(loc="upper left")
        ax1.grid(alpha=0.3)

        ax2.fill_between(dd.index, dd * 100, 0, color="#d62728", alpha=0.45, label="Drawdown")
        ax2.set_ylabel("Drawdown (%)")
        ax2.set_xlabel("Date")
        ax2.legend(loc="lower left")
        ax2.grid(alpha=0.3)

        plt.tight_layout()
        if save:
            out = Path(self.cfg.output_dir)
            out.mkdir(parents=True, exist_ok=True)
            path = out / "backtest_pnl.png"
            fig.savefig(path, dpi=150)
            print(f"[plot] Saved → {path}")
        plt.show()
        plt.close(fig)

    # ── save results ───────────────────────────────────────────────────────
    def save(self, results: dict) -> None:
        """Save daily_returns.csv and metrics_summary.csv to output_dir."""
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        agg = results.get("__aggregate__")
        if agg is None:
            print("[save] Nothing to save.")
            return

        ret_path = out / "daily_returns.csv"
        agg["daily_returns"].rename("daily_return").to_csv(ret_path, header=True)
        print(f"[save] Daily returns  → {ret_path}")

        rows = []
        for k, v in results.items():
            if "metrics" not in v:
                continue
            row = {"period": k}
            row.update(v["metrics"])
            rows.append(row)

        if rows:
            mpath = out / "metrics_summary.csv"
            pd.DataFrame(rows).set_index("period").to_csv(mpath)
            print(f"[save] Metrics        → {mpath}")


# ──────────────────────────────────────────────────────────
# CLI ENTRY-POINT
# ──────────────────────────────────────────────────────────
def main() -> None:
    import argparse

    parser = argparse.ArgumentParser(
        description="Pairs Trading Backtest Engine",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--prices", default=str(DEFAULT_CONFIG.engineered_features_path),
        help="Path to prices_features.csv",
    )
    parser.add_argument(
        "--pairs", default=str(DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv"),
        help="Path to discovered_pairs.csv",
    )
    parser.add_argument(
        "--model",
        default="zscore",
        help=(
            "Signal model to use. "
            "'zscore' (default) — classic fixed-threshold mean-reversion. "
            "'quantile_zscore' — per-pair adaptive thresholds from training "
            "z-score quantiles. "
            "Any other value is treated as a subdirectory name under "
            "data/processed/predictions/ (e.g. 'linear_regression', "
            "'xgboost_ols', 'xgboost_kalman', 'lstm_ols', 'lstm_kalman') and "
            "loads pre-computed ML predictions via PredictionSignal, which "
            "also uses per-pair quantile-calibrated thresholds."
        ),
    )
    # Fixed-threshold zscore params
    parser.add_argument("--entry_z",   type=float, default=2.0,  help="ZScore entry threshold (zscore model)")
    parser.add_argument("--exit_z",    type=float, default=0.5,  help="ZScore exit threshold (zscore model)")
    parser.add_argument("--stop_z",    type=float, default=4.0,  help="ZScore stop-loss, 0=off (zscore model)")
    # Quantile params — shared between quantile_zscore and ML PredictionSignal
    parser.add_argument(
        "--entry_q",
        type=float,
        default=DEFAULT_BACKTEST_PARAMS.entry_quantile,
        help="Entry quantile (lower tail) for quantile_zscore and ML models",
    )
    parser.add_argument(
        "--exit_q",
        type=float,
        default=DEFAULT_BACKTEST_PARAMS.exit_quantile,
        help="Exit quantile (lower tail) for quantile_zscore and ML models",
    )
    parser.add_argument(
        "--stop_q",
        type=float,
        default=DEFAULT_BACKTEST_PARAMS.stop_quantile,
        help="Stop-loss quantile, 0=off (quantile_zscore and ML models)",
    )
    parser.add_argument(
        "--pred_warmup",
        type=int,
        default=60,
        help="Days of test-period predictions used for per-pair PredictionSignal calibration",
    )
    parser.add_argument("--n_pairs",   type=int,   default=50,   help="Max pairs per window")
    parser.add_argument("--capital",   type=float, default=1e6,  help="Initial capital ($)")
    parser.add_argument("--tc_bps",    type=float, default=10.0, help="Transaction cost (bps/side)")
    parser.add_argument("--rolling_z", action="store_true",      help="Use rolling z-score")
    parser.add_argument("--lookback",  type=int,   default=63,   help="Rolling z-score lookback (days)")
    parser.add_argument("--holdout",   action="store_true",      help="Run holdout test (2017) only")
    parser.add_argument("--output",    default="outputs/backtest", help="Output directory")
    parser.add_argument("--save",      action="store_true",      help="Save results to CSV")
    parser.add_argument("--no_plot",   action="store_true",      help="Suppress plot")
    args = parser.parse_args()

    cfg = BacktestConfig(
        entry_z               = args.entry_z,
        exit_z                = args.exit_z,
        stop_z                = args.stop_z,
        n_top_pairs           = args.n_pairs,
        initial_capital       = args.capital,
        transaction_cost_bps  = args.tc_bps,
        use_rolling_zscore    = args.rolling_z,
        rolling_lookback      = args.lookback,
        output_dir            = args.output,
    )

    engine = BacktestEngine(cfg)
    engine.load_data(args.prices, args.pairs)

    model_name = args.model.lower()
    if model_name == "zscore":
        signal_generator = None  # engine falls back to ZScoreSignal from cfg
    elif model_name == "quantile_zscore":
        from src.backtest.quantile_zscore_signal import QuantileZScoreSignal
        signal_generator = QuantileZScoreSignal(
            entry_quantile   = args.entry_q,
            exit_quantile    = args.exit_q,
            stop_quantile    = args.stop_q,
            use_rolling      = args.rolling_z,
            rolling_lookback = args.lookback,
        )
        print(
            f"[BacktestEngine] Signal model   → QuantileZScoreSignal("
            f"entry_q={args.entry_q}, exit_q={args.exit_q}, stop_q={args.stop_q})"
        )
    else:
        predictions_root = DEFAULT_CONFIG.processed_dir / "predictions" / args.model
        if not predictions_root.exists():
            raise SystemExit(
                f"[error] Predictions directory not found: {predictions_root}\n"
                f"        Available models under data/processed/predictions/ "
                f"are the subdirectory names there."
            )
        signal_generator = PredictionSignal(
            predictions_root = predictions_root,
            entry_quantile   = args.entry_q,
            exit_quantile    = args.exit_q,
            stop_quantile    = args.stop_q,
            warmup_days      = args.pred_warmup,
        )
        print(
            f"[BacktestEngine] Signal model   → PredictionSignal({args.model}, "
            f"entry_q={args.entry_q}, exit_q={args.exit_q}, "
            f"stop_q={args.stop_q}, warmup={args.pred_warmup}d)"
        )

    if args.holdout:
        results = engine.run_holdout(signal_generator=signal_generator)
    else:
        results = engine.run(signal_generator=signal_generator)

    if isinstance(signal_generator, PredictionSignal):
        print(
            f"  → matched predictions for {signal_generator.n_matched_pairs} pair-windows, "
            f"missing for {signal_generator.n_missing_pairs}"
        )

    engine.report(results)

    if args.save:
        engine.save(results)
    if not args.no_plot:
        engine.plot(results)


if __name__ == "__main__":
    main()
