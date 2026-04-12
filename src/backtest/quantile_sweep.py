#!/usr/bin/env python3
"""
Quantile Z-Score Sweep
======================
Grid-search over quantile levels for QuantileZScoreSignal using the same
expanding-window cross-validation as backtest_engine.py.  Compares quantile-
based thresholds against the fixed-threshold ZScoreSignal baseline.

Outputs
-------
  outputs/quantile_sweep/
    sweep_results.csv       – full grid of (entry_q, exit_q, stop_q) → metrics
    comparison_table.csv    – side-by-side: fixed baseline vs best quantile
    quantile_sweep_heatmap.png  – Sharpe heatmap (entry_q × exit_q)
    quantile_vs_fixed.png       – bar chart comparison

Usage
-----
    python -m src.backtest.quantile_sweep
    python -m src.backtest.quantile_sweep --entry_qs 0.03 0.05 0.07 0.10
    python -m src.backtest.quantile_sweep --no_plot
"""

from __future__ import annotations

import argparse
import itertools
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

from src.backtest.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    ZScoreSignal,
)
from src.backtest.quantile_zscore_signal import QuantileZScoreSignal
from src.config import DEFAULT_BACKTEST_PARAMS, DEFAULT_CONFIG

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────────────────
# DEFAULT GRID
# ──────────────────────────────────────────────────────────
DEFAULT_ENTRY_QS = list(DEFAULT_BACKTEST_PARAMS.sweep_entry_qs)
DEFAULT_EXIT_QS  = list(DEFAULT_BACKTEST_PARAMS.sweep_exit_qs)
DEFAULT_STOP_QS  = list(DEFAULT_BACKTEST_PARAMS.sweep_stop_qs)

# Fixed-threshold baselines to compare against (sourced from config defaults)
FIXED_BASELINES = [
    {
        "entry_z": DEFAULT_BACKTEST_PARAMS.entry_z,
        "exit_z":  DEFAULT_BACKTEST_PARAMS.exit_z,
        "stop_z":  DEFAULT_BACKTEST_PARAMS.stop_z,
    },
]


def _run_one(
    engine: BacktestEngine,
    label: str,
    signal_generator,
) -> dict:
    """Run expanding-window CV and return aggregate metrics + label."""
    results = engine.run(signal_generator=signal_generator)
    agg = results.get("__aggregate__", {})
    metrics = agg.get("metrics", {})
    metrics["label"] = label
    # also store per-fold Sharpe for robustness check
    fold_sharpes = []
    for k, v in results.items():
        if k == "__aggregate__":
            continue
        fold_sharpes.append(v.get("metrics", {}).get("sharpe", 0.0))
    metrics["fold_sharpes"] = fold_sharpes
    metrics["mean_fold_sharpe"] = float(np.mean(fold_sharpes)) if fold_sharpes else 0.0
    metrics["std_fold_sharpe"] = float(np.std(fold_sharpes)) if fold_sharpes else 0.0
    return metrics


def run_sweep(
    engine: BacktestEngine,
    entry_qs: list[float],
    exit_qs: list[float],
    stop_qs: list[float],
    use_rolling: bool = False,
    rolling_lookback: int = DEFAULT_BACKTEST_PARAMS.rolling_lookback,
) -> pd.DataFrame:
    """
    Grid-search over (entry_q, exit_q, stop_q) combinations.

    Returns DataFrame with one row per configuration, columns include
    all metrics from compute_metrics plus quantile parameters.
    """
    rows: list[dict] = []
    combos = list(itertools.product(entry_qs, exit_qs, stop_qs))
    n_total = len(combos)

    print(f"\n{'='*72}")
    print(f"  QUANTILE SWEEP  ({n_total} configurations)")
    print(f"{'='*72}\n")

    for idx, (eq, xq, sq) in enumerate(combos, 1):
        if xq >= (1 - eq):
            # exit quantile too wide relative to entry — skip invalid
            continue

        label = f"Q(e={eq:.2f},x={xq:.2f},s={sq:.2f})"
        print(f"[{idx:3d}/{n_total}] {label}")

        sig = QuantileZScoreSignal(
            entry_quantile=eq,
            exit_quantile=xq,
            stop_quantile=sq,
            use_rolling=use_rolling,
            rolling_lookback=rolling_lookback,
        )
        m = _run_one(engine, label, sig)
        m["entry_quantile"] = eq
        m["exit_quantile"] = xq
        m["stop_quantile"] = sq
        m["type"] = "quantile"
        rows.append(m)

    return pd.DataFrame(rows)


def run_fixed_baselines(engine: BacktestEngine) -> pd.DataFrame:
    """Run the fixed-threshold baseline(s) for comparison."""
    rows: list[dict] = []

    print(f"\n{'='*72}")
    print(f"  FIXED BASELINES  ({len(FIXED_BASELINES)} configurations)")
    print(f"{'='*72}\n")

    for params in FIXED_BASELINES:
        label = f"Fixed(e={params['entry_z']},x={params['exit_z']},s={params['stop_z']})"
        print(f"  {label}")

        sig = ZScoreSignal(
            entry_z=params["entry_z"],
            exit_z=params["exit_z"],
            stop_z=params["stop_z"],
        )
        m = _run_one(engine, label, sig)
        m["entry_z_fixed"] = params["entry_z"]
        m["exit_z_fixed"] = params["exit_z"]
        m["stop_z_fixed"] = params["stop_z"]
        m["type"] = "fixed"
        rows.append(m)

    return pd.DataFrame(rows)


# ──────────────────────────────────────────────────────────
# COMPARISON TABLE
# ──────────────────────────────────────────────────────────
REPORT_COLS = [
    "label", "type",
    "sharpe", "fitness", "annualized_return", "max_drawdown",
    "volatility", "turnover", "margin_permille", "n_trades",
    "mean_fold_sharpe", "std_fold_sharpe",
]


def build_comparison(
    sweep_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    rank_by: str = "sharpe",
    top_n: int = 5,
) -> pd.DataFrame:
    """
    Build a comparison table: fixed baseline vs top quantile configs.

    Returns the fixed row(s) + top-N quantile rows, sorted by rank_by.
    """
    top_quantile = (
        sweep_df
        .sort_values(rank_by, ascending=False)
        .head(top_n)
    )
    combined = pd.concat([fixed_df, top_quantile], ignore_index=True)
    cols = [c for c in REPORT_COLS if c in combined.columns]
    return combined[cols].sort_values(rank_by, ascending=False).reset_index(drop=True)


# ──────────────────────────────────────────────────────────
# PLOTS
# ──────────────────────────────────────────────────────────
def plot_heatmap(sweep_df: pd.DataFrame, out_dir: Path) -> None:
    """Sharpe heatmap: entry_quantile (y) × exit_quantile (x)."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping heatmap.")
        return

    if sweep_df.empty:
        return

    # Average over stop_quantile for heatmap
    pivot = (
        sweep_df
        .groupby(["entry_quantile", "exit_quantile"])["sharpe"]
        .mean()
        .unstack()
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    im = ax.imshow(
        pivot.values,
        aspect="auto",
        cmap="RdYlGn",
        origin="lower",
    )
    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"{v:.2f}" for v in pivot.columns])
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"{v:.2f}" for v in pivot.index])
    ax.set_xlabel("Exit Quantile")
    ax.set_ylabel("Entry Quantile")
    ax.set_title("Aggregate Sharpe — Quantile Z-Score Sweep")

    # annotate cells
    for i in range(len(pivot.index)):
        for j in range(len(pivot.columns)):
            val = pivot.values[i, j]
            if not np.isnan(val):
                ax.text(j, i, f"{val:.2f}", ha="center", va="center", fontsize=10)

    fig.colorbar(im, ax=ax, label="Sharpe Ratio")
    plt.tight_layout()

    path = out_dir / "quantile_sweep_heatmap.png"
    fig.savefig(path, dpi=150)
    print(f"[plot] Heatmap saved → {path}")
    plt.close(fig)


def plot_comparison_bar(
    comparison_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Bar chart: Sharpe and Fitness for fixed vs. top quantile configs."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        print("[plot] matplotlib not installed — skipping bar chart.")
        return

    if comparison_df.empty:
        return

    labels = comparison_df["label"].tolist()
    x = np.arange(len(labels))
    width = 0.35

    fig, ax1 = plt.subplots(figsize=(12, 6))

    sharpes = comparison_df["sharpe"].values
    fitnesses = comparison_df["fitness"].values

    bars1 = ax1.bar(x - width / 2, sharpes, width, label="Sharpe", color="#1f77b4")
    bars2 = ax1.bar(x + width / 2, fitnesses, width, label="Fitness", color="#ff7f0e")

    ax1.set_ylabel("Metric Value")
    ax1.set_title("Fixed vs. Quantile Thresholds — Sharpe & Fitness")
    ax1.set_xticks(x)
    ax1.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
    ax1.legend()
    ax1.grid(axis="y", alpha=0.3)
    ax1.axhline(0, color="black", lw=0.7, ls="--")

    # annotate bars
    for bar in bars1:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        h = bar.get_height()
        ax1.annotate(f"{h:.2f}", xy=(bar.get_x() + bar.get_width() / 2, h),
                     xytext=(0, 3), textcoords="offset points",
                     ha="center", va="bottom", fontsize=8)

    plt.tight_layout()

    path = out_dir / "quantile_vs_fixed.png"
    fig.savefig(path, dpi=150)
    print(f"[plot] Comparison bar chart saved → {path}")
    plt.close(fig)


def plot_fold_robustness(
    sweep_df: pd.DataFrame,
    fixed_df: pd.DataFrame,
    out_dir: Path,
) -> None:
    """Box plot of per-fold Sharpe ratios for top configs vs fixed baseline."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        return

    if sweep_df.empty:
        return

    # Top quantile configs by aggregate Sharpe (match comparison table count)
    top_n = min(5, len(sweep_df))
    top_q = sweep_df.sort_values("sharpe", ascending=False).head(top_n)
    all_configs = pd.concat([fixed_df, top_q], ignore_index=True)

    fig, ax = plt.subplots(figsize=(12, 5))
    data = []
    labels = []
    for _, row in all_configs.iterrows():
        fold_s = row.get("fold_sharpes", [])
        if fold_s:
            data.append(fold_s)
            labels.append(row["label"])

    if data:
        bp = ax.boxplot(data, labels=labels, patch_artist=True)
        colors = ["#d62728"] * len(fixed_df) + ["#1f77b4"] * len(top_q)
        for patch, color in zip(bp["boxes"], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.6)
        ax.set_ylabel("Fold Sharpe Ratio")
        ax.set_title("Per-Fold Sharpe Robustness — Fixed (red) vs. Quantile (blue)")
        ax.set_xticklabels(labels, rotation=30, ha="right", fontsize=8)
        ax.axhline(0, color="black", lw=0.7, ls="--")
        ax.grid(axis="y", alpha=0.3)
        plt.tight_layout()

        path = out_dir / "fold_robustness.png"
        fig.savefig(path, dpi=150)
        print(f"[plot] Fold robustness saved → {path}")
    plt.close(fig)


# ──────────────────────────────────────────────────────────
# REPORT PRINTER
# ──────────────────────────────────────────────────────────
def print_comparison(comp_df: pd.DataFrame) -> None:
    """Pretty-print the comparison table to stdout."""
    print(f"\n{'='*90}")
    print("  COMPARISON: Fixed Thresholds vs. Best Quantile Configs")
    print(f"{'='*90}")

    header = (
        f"  {'Config':<42} {'Sharpe':>7} {'Fitness':>8} {'Return':>8}"
        f" {'MaxDD':>8} {'Turnover':>9} {'Margin':>8} {'Trades':>7}"
    )
    print(header)
    print("  " + "─" * 86)

    for _, row in comp_df.iterrows():
        label = row.get("label", "?")
        print(
            f"  {label:<42}"
            f" {row.get('sharpe', 0):>7.2f}"
            f" {row.get('fitness', 0):>8.2f}"
            f" {row.get('annualized_return', 0):>7.2%}"
            f" {row.get('max_drawdown', 0):>7.2%}"
            f" {row.get('turnover', 0):>8.2%}"
            f" {row.get('margin_permille', 0):>6.2f}‰"
            f" {int(row.get('n_trades', 0)):>7,}"
        )

    print(f"  {'='*86}\n")


# ──────────────────────────────────────────────────────────
# MAIN
# ──────────────────────────────────────────────────────────
def main() -> None:
    parser = argparse.ArgumentParser(
        description="Quantile Z-Score Sweep — grid search over quantile thresholds",
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
        "--entry_qs", nargs="+", type=float, default=DEFAULT_ENTRY_QS,
        help="Entry quantile levels to sweep",
    )
    parser.add_argument(
        "--exit_qs", nargs="+", type=float, default=DEFAULT_EXIT_QS,
        help="Exit quantile levels to sweep",
    )
    parser.add_argument(
        "--stop_qs", nargs="+", type=float, default=DEFAULT_STOP_QS,
        help="Stop-loss quantile levels to sweep",
    )
    parser.add_argument("--n_pairs", type=int, default=DEFAULT_BACKTEST_PARAMS.n_top_pairs, help="Max pairs per window")
    parser.add_argument("--capital", type=float, default=DEFAULT_BACKTEST_PARAMS.initial_capital, help="Initial capital ($)")
    parser.add_argument("--tc_bps", type=float, default=DEFAULT_BACKTEST_PARAMS.transaction_cost_bps, help="Transaction cost (bps)")
    parser.add_argument("--rolling_z", action="store_true", help="Use rolling z-score")
    parser.add_argument("--lookback", type=int, default=DEFAULT_BACKTEST_PARAMS.rolling_lookback, help="Rolling z-score lookback")
    parser.add_argument("--output", default=DEFAULT_BACKTEST_PARAMS.sweep_output_dir, help="Output directory")
    parser.add_argument("--no_plot", action="store_true", help="Suppress plots")
    parser.add_argument("--top_n", type=int, default=DEFAULT_BACKTEST_PARAMS.sweep_top_n, help="Number of top quantile configs in comparison table")
    parser.add_argument(
        "--rank_by", default="sharpe", choices=["sharpe", "fitness", "annualized_return"],
        help="Metric to rank configs by",
    )
    args = parser.parse_args()

    # ── setup engine ─────────────────────────────────────────
    cfg = BacktestConfig(
        n_top_pairs=args.n_pairs,
        initial_capital=args.capital,
        transaction_cost_bps=args.tc_bps,
        use_rolling_zscore=args.rolling_z,
        rolling_lookback=args.lookback,
        output_dir=args.output,
    )
    engine = BacktestEngine(cfg)
    engine.load_data(args.prices, args.pairs)

    # ── run fixed baseline ───────────────────────────────────
    fixed_df = run_fixed_baselines(engine)

    # ── run quantile sweep ───────────────────────────────────
    sweep_df = run_sweep(
        engine,
        entry_qs=args.entry_qs,
        exit_qs=args.exit_qs,
        stop_qs=args.stop_qs,
        use_rolling=args.rolling_z,
        rolling_lookback=args.lookback,
    )

    # ── comparison ───────────────────────────────────────────
    comp_df = build_comparison(sweep_df, fixed_df, rank_by=args.rank_by, top_n=args.top_n)
    print_comparison(comp_df)

    # ── save ─────────────────────────────────────────────────
    out_dir = Path(args.output)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not sweep_df.empty:
        # drop fold_sharpes list column for CSV
        save_cols = [c for c in sweep_df.columns if c != "fold_sharpes"]
        sweep_df[save_cols].to_csv(out_dir / "sweep_results.csv", index=False)
        print(f"[save] Sweep results  → {out_dir / 'sweep_results.csv'}")

    comp_save = comp_df.drop(columns=["fold_sharpes"], errors="ignore")
    comp_save.to_csv(out_dir / "comparison_table.csv", index=False)
    print(f"[save] Comparison     → {out_dir / 'comparison_table.csv'}")

    # ── plots ────────────────────────────────────────────────
    if not args.no_plot:
        plot_heatmap(sweep_df, out_dir)
        plot_comparison_bar(comp_df, out_dir)
        plot_fold_robustness(sweep_df, fixed_df, out_dir)

    print("\nDone.")


if __name__ == "__main__":
    main()
