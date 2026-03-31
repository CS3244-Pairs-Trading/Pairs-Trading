#!/usr/bin/env python3
"""
Pairs Trading Backtest Engine
==============================
Evaluates pairs discovered by rank_pairs.py using a z-score
mean-reversion strategy across expanding validation windows.

Metrics
-----------------------------------------
  Sharpe   – annualised risk-adjusted return
  Turnover – annualised fraction of book traded
  Fitness  – Sharpe × √( |Returns| / max(Turnover, 12.5%) )
  Margin   – ( Returns / Turnover ) × 20  [per-mille, ‰]
  Returns  – annualised
  Drawdown – maximum peak-to-trough

Quick-start (programmatic)
---------------------------
    from src.backtest.backtest_engine import BacktestEngine, BacktestConfig

    engine = BacktestEngine(BacktestConfig(entry_z=2.0, exit_z=0.5))
    engine.load_data(
        "data/processed/prices_features.csv",
        "data/processed/discovered_pairs.csv",
    )
    results = engine.run()            # expanding-window CV  (2013–2016)
    results = engine.run_holdout()    # final holdout test   (2017)
    engine.report(results)
    engine.plot(results)
    engine.save(results)              # writes CSVs to outputs/backtest/

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
from typing import Optional
import numpy as np
import pandas as pd
from src.config import DEFAULT_CONFIG
warnings.filterwarnings("ignore")

# CONSTANTS
TRADING_DAYS    = 252
RISK_FREE_RATE  = 0.0        # annualised; set to e.g. 0.04 for a 4 % hurdle
FITNESS_TV_FLOOR = 0.125 

# CONFIGURATION
# Possible for tuning based on strategy
@dataclass
class BacktestConfig:
    """All tuneable parameters for the backtest."""

    # strategy signals
    entry_z: float = 2.0   # open a trade when |z| crosses this
    exit_z:  float = 0.5   # close a trade when |z| falls below this
    stop_z:  float = 4.0   # emergency stop-loss threshold (0 = disabled)

    # z-score computation
    use_rolling_zscore: bool = False   # True → rolling window; False → fixed training stats
    rolling_lookback:   int  = 63      # rolling window in trading days (≈ 3 months)

    # capital & sizing
    initial_capital: float = 1_000_000.0
    n_top_pairs:     int   = 50   # max pairs selected per training window

    # cost model
    transaction_cost_bps: float = 10.0   # basis points per side per trade leg

    # pair selection
    min_score: float = 0.0   # discard pairs below this composite score

    # output
    output_dir: str = "outputs/backtest"


# PERFORMANCE METRICS
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
    daily_turnover : fraction of book traded each day  (decimal, one-sided)
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

    # Margin[‰]  =  (ann_return / ann_turnover) × 20
    # (Returns/Turnover) × 20 already yields per-mille
    margin = (ann_return / ann_tv * 20.0) if ann_tv > 0 else 0.0

    # Fitness  =  Sharpe × √(|ann_return| / max(ann_turnover, 12.5%))
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


# SINGLE-PAIR SIMULATION
def _spread_stats(c1_train: pd.Series, c2_train: pd.Series, beta: float) -> dict:
    """Compute spread mean / std on training data only (no look-ahead)."""
    spread = c1_train - beta * c2_train
    return {
        "mean": float(spread.mean()),
        "std":  float(spread.std()),
        "beta": beta,
    }


def _zscore(
    spread:       pd.Series,
    stats:        dict,
    use_rolling:  bool,
    lookback:     int,
) -> pd.Series:
    """Return z-score series; uses fixed training stats or rolling window."""
    if use_rolling:
        min_p = max(1, lookback // 2)
        mu    = spread.rolling(lookback, min_periods=min_p).mean()
        sig   = spread.rolling(lookback, min_periods=min_p).std().clip(lower=1e-8)
    else:
        mu  = stats["mean"]
        sig = max(stats["std"], 1e-8)
    return (spread - mu) / sig


def simulate_pair(
    c1:         pd.Series,
    c2:         pd.Series,
    stats:      dict,
    cfg:        BacktestConfig,
    allocation: float,
) -> tuple[pd.Series, pd.Series, dict, dict]:
    """
    Simulate a single pair over the test period.

    Spread definition  (matching rank_pairs.py):
        spread = c1 − beta × c2

    Position sizing (beta-adjusted, dollar-neutral):
        long  leg  →  k1 = allocation / (1 + |beta|)    dollars in  stock1
        short leg  →  k2 = k1 × |beta|                  dollars in  stock2
        total book                                       = allocation  (always)

    Signals:
        z < −entry_z  →  long  spread (+1): long  c1, short c2
        z >  entry_z  →  short spread (−1): short c1, long  c2
        |z| < exit_z  →  flat  (0)
        |z| > stop_z  →  flat  (emergency stop, if stop_z > 0)

    Parameters
    ----------
    c1, c2     : aligned Close price series for the test period
    stats      : dict with 'mean', 'std', 'beta'  from training data
    cfg        : BacktestConfig
    allocation : dollar allocation for this pair

    Returns
    -------
    daily_pnl    : pd.Series  daily P&L in dollars
    daily_tv     : pd.Series  dollars traded each day (for turnover)
    n_long_yr    : dict {year → count of long-spread entries}
    n_short_yr   : dict {year → count of short-spread entries}
    """
    beta    = stats["beta"]
    denom   = 1.0 + abs(beta)
    k1      = allocation / denom               # dollars in stock1 leg
    k2      = allocation * abs(beta) / denom   # dollars in stock2 leg
    tc_rate = cfg.transaction_cost_bps / 10_000.0

    spread = c1 - beta * c2
    z      = _zscore(spread, stats, cfg.use_rolling_zscore, cfg.rolling_lookback)
    ret1   = c1.pct_change().fillna(0.0)
    ret2   = c2.pct_change().fillna(0.0)

    n        = len(c1)
    pnl_arr  = np.zeros(n)
    tv_arr   = np.zeros(n)
    pos      = 0       # +1 long spread, −1 short spread, 0 flat

    n_long_yr:  dict[int, int] = {}
    n_short_yr: dict[int, int] = {}

    for i in range(1, n):
        zi = float(z.iloc[i])
        if np.isnan(zi):
            continue

        new_pos = pos
        traded  = False

        # signal logic
        if pos == 0:
            if zi <= -cfg.entry_z:
                new_pos, traded = 1,  True    # enter long spread
            elif zi >= cfg.entry_z:
                new_pos, traded = -1, True    # enter short spread

        else:
            if abs(zi) <= cfg.exit_z:
                new_pos, traded = 0, True     # normal exit
            elif cfg.stop_z > 0 and abs(zi) >= cfg.stop_z:
                new_pos, traded = 0, True     # stop-loss exit
            elif pos == 1 and zi >= cfg.entry_z:
                new_pos, traded = -1, True    # reverse: long → short
            elif pos == -1 and zi <= -cfg.entry_z:
                new_pos, traded = 1,  True    # reverse: short → long

        # daily P&L for the position held into day i
        # long  spread (+1): long k1 in c1, short k2 in c2
        # short spread (−1): short k1 in c1, long  k2 in c2
        if pos != 0:
            pnl_arr[i] = pos * (
                k1 * float(ret1.iloc[i]) - k2 * float(ret2.iloc[i])
            )

        # transaction costs & turnover
        if traded and new_pos != pos:
            # |change| = 1 → open or close one spread unit
            # |change| = 2 → reverse (close old + open new)
            delta          = abs(new_pos - pos)
            dollars_traded = delta * allocation      # total book change
            # cost: tc_rate per side × 2 legs per spread × delta units
            cost           = dollars_traded * tc_rate * 2.0
            pnl_arr[i]    -= cost
            tv_arr[i]      = dollars_traded

            # trade bookkeeping
            yr = int(c1.index[i].year)
            if new_pos == 1 or (new_pos == 0 and pos == -1):
                # opened long OR closed a short
                n_long_yr[yr] = n_long_yr.get(yr, 0) + 1
            if new_pos == -1 or (new_pos == 0 and pos == 1):
                # opened short OR closed a long
                n_short_yr[yr] = n_short_yr.get(yr, 0) + 1

        pos = new_pos

    # force-close any open position at period end
    if pos != 0:
        yr = int(c1.index[-1].year)
        if pos == 1:
            n_short_yr[yr] = n_short_yr.get(yr, 0) + 1   # forced close of long
        else:
            n_long_yr[yr]  = n_long_yr.get(yr, 0)  + 1   # forced close of short

    idx = c1.index
    return (
        pd.Series(pnl_arr, index=idx, name="pnl"),
        pd.Series(tv_arr,  index=idx, name="tv"),
        n_long_yr,
        n_short_yr,
    )


# BACKTEST ENGINE
class BacktestEngine:
    """
    End-to-end backtesting for the pairs trading strategy.

    The engine consumes two artefacts produced by the upstream pipeline:
      * prices_features.csv  – cleaned daily OHLCV + SimpleReturn + LogPrice
      * discovered_pairs.csv – ranked pairs with training-window metadata

    For each training window it selects the top-N eligible pairs by score,
    computes spread statistics on the training data (no look-ahead), and
    simulates the z-score strategy on the out-of-sample validation period.

    See module docstring for usage examples.
    """

    def __init__(self, config: Optional[BacktestConfig] = None) -> None:
        self.cfg   = config or BacktestConfig()
        self._wide: Optional[pd.DataFrame] = None   # wide Close  (Date × Ticker)
        self.pairs: Optional[pd.DataFrame] = None   # discovered_pairs.csv

    # data loading
    def load_data(
        self,
        prices_path: str | Path = DEFAULT_CONFIG.engineered_features_path,
        pairs_path:  str | Path = DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv",
    ) -> "BacktestEngine":
        """
        Load price data and discovered pairs into the engine.

        Parameters
        ----------
        prices_path : path to prices_features.csv  (long format, must have
                      columns: Date, Ticker, Close)
        pairs_path  : path to discovered_pairs.csv  (output of rank_pairs.py)

        Returns
        -------
        self  (for chaining)
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
        self.pairs  = pd.read_csv(pairs_path)
        n_eligible  = int(self.pairs["is_eligible"].sum())
        print(f"  → {len(self.pairs):,} total pairs | {n_eligible:,} eligible\n")

        return self

    # single-window backtest
    def _run_window(
        self,
        window_label: str,
        train_end:    str,
        test_start:   str,
        test_end:     str,
    ) -> Optional[dict]:
        """
        Backtest one train/test split.

        Returns None when no eligible pairs or price data are found.
        """
        cfg = self.cfg

        #1. Select top pairs
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

        #2. Slice price matrices for training stats and test simulation
        train_close = self._wide.loc[:train_end]
        test_close  = self._wide.loc[test_start:test_end]

        if test_close.empty:
            print(f"  [SKIP] {window_label}: no price data for {test_start}–{test_end}")
            return None

        n_pairs    = len(window_pairs)
        allocation = cfg.initial_capital / max(n_pairs, 1)

        print(
            f"[{window_label}]  pairs={n_pairs:3d}  |  "
            f"test {test_start} → {test_end}  ({len(test_close)} days)"
        )

        #3. Simulate
        all_pnl:     dict[str, pd.Series] = {}
        all_tv:      dict[str, pd.Series] = {}
        n_long_yr:   dict[int, int]       = {}
        n_short_yr:  dict[int, int]       = {}

        for _, row in window_pairs.iterrows():
            pair_name = row["pair"]
            s1, s2    = pair_name.split("-", 1)
            beta      = float(row["initial_beta"])

            if s1 not in self._wide.columns or s2 not in self._wide.columns:
                continue

            # training alignment (spread stats only)
            tr1   = train_close[s1].dropna()
            tr2   = train_close[s2].dropna()
            tr_idx = tr1.index.intersection(tr2.index)
            if len(tr_idx) < 63:      # need ≥ 3 months to estimate stats
                continue
            s_stats = _spread_stats(tr1.loc[tr_idx], tr2.loc[tr_idx], beta)

            # test alignment
            te1    = test_close[s1].dropna()
            te2    = test_close[s2].dropna()
            te_idx = te1.index.intersection(te2.index)
            if len(te_idx) < 5:
                continue

            pnl, tv, lng, sht = simulate_pair(
                te1.loc[te_idx], te2.loc[te_idx], s_stats, cfg, allocation
            )
            all_pnl[pair_name] = pnl
            all_tv[pair_name]  = tv

            for yr, cnt in lng.items():
                n_long_yr[yr]  = n_long_yr.get(yr, 0)  + cnt
            for yr, cnt in sht.items():
                n_short_yr[yr] = n_short_yr.get(yr, 0) + cnt

        if not all_pnl:
            print(f"  [SKIP] {window_label}: no pairs produced P&L")
            return None

        # 4. Aggregate to portfolio
        pnl_df = pd.DataFrame(all_pnl).fillna(0.0)
        tv_df  = pd.DataFrame(all_tv).fillna(0.0)

        port_pnl = pnl_df.sum(axis=1)
        port_tv  = tv_df.sum(axis=1)

        daily_ret = port_pnl / cfg.initial_capital
        daily_tv  = port_tv  / cfg.initial_capital   # fraction of book

        n_trades = sum(n_long_yr.values()) + sum(n_short_yr.values())
        metrics  = compute_metrics(daily_ret, n_trades, daily_tv)

        return {
            "window":         window_label,
            "test_start":     test_start,
            "test_end":       test_end,
            "n_pairs":        n_pairs,
            "daily_returns":  daily_ret,
            "daily_turnover": daily_tv,
            "daily_pnl":      port_pnl,
            "n_long_yr":      n_long_yr,
            "n_short_yr":     n_short_yr,
            "n_trades_total": n_trades,
            "metrics":        metrics,
        }

    # expanding-window cross-validation
    def run(
        self,
        windows: Optional[list[str]] = None,
    ) -> dict:
        """
        Run expanding-window out-of-sample backtest using the CV folds
        defined in src/config.py.

        Each window's pairs were discovered only on training data and are
        evaluated on the following out-of-sample validation year → no leakage.

        Parameters
        windows : window labels to run (default: all 4 CV folds).
                  Example: ["2010_2012", "2010_2013", "2010_2014", "2010_2015"]

        Returns
        dict keyed by window label + "__aggregate__"
        Each value contains: window, test_start, test_end, n_pairs,
        daily_returns, daily_turnover, n_long_yr, n_short_yr, metrics
        """
        if self.pairs is None or self._wide is None:
            raise RuntimeError("Call load_data() before run().")

        # build window → (train_end, test_start, test_end) map from config
        cfg       = DEFAULT_CONFIG
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
            res = self._run_window(w, train_end, test_start, test_end)
            if res:
                results[w] = res

        if results:
            results["__aggregate__"] = self._aggregate(results)

        return results

    # final holdout test
    def run_holdout(self) -> dict:
        """
        Run the final holdout test (train 2010–2016, test 2017).

        Uses pairs discovered under window label "2010_2016".
        Ensure rank_pairs.py has been run with the holdout training window
        before calling this method.
        """
        if self.pairs is None or self._wide is None:
            raise RuntimeError("Call load_data() before run_holdout().")

        hs  = DEFAULT_CONFIG.holdout_split
        res = self._run_window(hs.label, hs.train.end, hs.test.start, hs.test.end)
        if not res:
            return {}

        return {hs.label: res, "__aggregate__": res}

    #internal: aggregate helper
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
            "daily_returns":   all_ret,
            "daily_turnover":  all_tv,
            "n_long_total":    n_long,
            "n_short_total":   n_short,
            "n_trades_total":  n_long + n_short,
            "metrics":         compute_metrics(all_ret, n_long + n_short, all_tv),
        }

    #reporting
    def report(self, results: dict, by_year: bool = True) -> None:
        """
        Columns match the reference screenshots:
            Year | Sharpe | Turnover | Fitness | Returns | Drawdown | Margin | Long | Short
        """
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

        #per-year breakdown
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

        header = (
            f"\n  {'Year':<6} {'Sharpe':>7} {'Turnover':>10} {'Fitness':>8}"
            f" {'Returns':>9} {'Drawdown':>10} {'Margin':>10}"
            f" {'Long':>6} {'Short':>6}"
        )
        print(header)
        print("  " + "─" * 78)

        for year, grp in dr.groupby(dr.index.year):
            tv_g  = dt.reindex(grp.index).fillna(0.0)
            lc    = lng_yr.get(year, 0)
            sc    = sht_yr.get(year, 0)
            ym    = compute_metrics(grp, lc + sc, tv_g)
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

    #plot
    def plot(self, results: dict, save: bool = True) -> None:
        """
        Plot cumulative-return curve and drawdown for the aggregate period.

        Output  →  <BacktestConfig.output_dir>/backtest_pnl.png
        """
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

    #save results
    def save(self, results: dict) -> None:
        """
        Save daily returns and metrics summary as CSV files.

        Output files:
            <output_dir>/daily_returns.csv
            <output_dir>/metrics_summary.csv
        """
        out = Path(self.cfg.output_dir)
        out.mkdir(parents=True, exist_ok=True)

        agg = results.get("__aggregate__")
        if agg is None:
            print("[save] Nothing to save.")
            return

        # daily returns
        ret_path = out / "daily_returns.csv"
        agg["daily_returns"].rename("daily_return").to_csv(ret_path, header=True)
        print(f"[save] Daily returns  → {ret_path}")

        # per-window + aggregate metrics
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


# CLI ENTRY-POINT

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
    parser.add_argument("--entry_z",   type=float, default=2.0,  help="Entry z-score threshold")
    parser.add_argument("--exit_z",    type=float, default=0.5,  help="Exit z-score threshold")
    parser.add_argument("--stop_z",    type=float, default=4.0,  help="Stop-loss z-score (0=off)")
    parser.add_argument("--n_pairs",   type=int,   default=50,   help="Max pairs per window")
    parser.add_argument("--capital",   type=float, default=1e6,  help="Initial capital ($)")
    parser.add_argument("--tc_bps",    type=float, default=10.0, help="Transaction cost (bps/side)")
    parser.add_argument("--rolling_z", action="store_true",      help="Use rolling z-score instead of fixed training stats")
    parser.add_argument("--lookback",  type=int,   default=63,   help="Rolling z-score lookback (trading days)")
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

    results = engine.run_holdout() if args.holdout else engine.run()
    engine.report(results)

    if args.save:
        engine.save(results)

    if not args.no_plot:
        engine.plot(results)


if __name__ == "__main__":
    main()
