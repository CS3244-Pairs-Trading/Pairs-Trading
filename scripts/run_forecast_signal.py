#!/usr/bin/env python3
"""
Run the forecast-based strategy with causal calibration and qualification.

Validation:
    - evaluates decile / quintile thresholds per pair
    - ignores inactive 0-trade rows when selecting a global default mode

Holdout:
    - calibrates thresholds from an early warmup slice of 2017
    - runs a short qualification window to choose active pairs and per-pair modes
    - deploys capital only after qualification on the selected live pairs
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from src.backtest.backtest_engine import (
    BacktestConfig,
    BacktestEngine,
    compute_metrics,
    execute_signals,
)
from src.backtest.forecast_signal import ForecastSignal
from src.config import DEFAULT_CONFIG


def _spread_stats(c1_train: pd.Series, c2_train: pd.Series, beta: float) -> dict:
    spread = c1_train - beta * c2_train
    return {
        "mean": float(spread.mean()),
        "std": float(spread.std()),
        "beta": beta,
    }


def _build_pair_job(
    row: pd.Series,
    train_close: pd.DataFrame,
    test_close: pd.DataFrame,
) -> Optional[dict]:
    s1 = row["stock_a"]
    s2 = row["stock_b"]
    if s1 not in train_close.columns or s2 not in train_close.columns:
        return None
    if s1 not in test_close.columns or s2 not in test_close.columns:
        return None

    tr1 = train_close[s1].dropna()
    tr2 = train_close[s2].dropna()
    tr_idx = tr1.index.intersection(tr2.index)
    if len(tr_idx) < 63:
        return None

    te1 = test_close[s1].dropna()
    te2 = test_close[s2].dropna()
    te_idx = te1.index.intersection(te2.index)
    if len(te_idx) < 5:
        return None

    return {
        "pair": row["pair"],
        "stock_a": s1,
        "stock_b": s2,
        "beta": float(row["initial_beta"]),
        "tr1": tr1.loc[tr_idx],
        "tr2": tr2.loc[tr_idx],
        "te1": te1.loc[te_idx],
        "te2": te2.loc[te_idx],
    }


def _simulate_pair_mode(
    job: dict,
    predictions_root: Path,
    window_label: str,
    cfg: BacktestConfig,
    horizon: int,
    mode: str,
    warmup_days: int,
    allocation: float,
    active_start: int = 0,
    active_end: int | None = None,
) -> dict:
    signal = ForecastSignal(
        predictions_root,
        horizon=horizon,
        threshold_mode=mode,
        warmup_days=warmup_days,
    )
    stats = _spread_stats(job["tr1"], job["tr2"], job["beta"])
    stats["pair"] = job["pair"]
    stats["window"] = window_label

    signal.fit(job["tr1"], job["tr2"], stats)
    signals = signal.predict(job["te1"], job["te2"])
    beta_exec = signal.get_execution_beta()
    if beta_exec is None:
        beta_exec = job["beta"]

    exec_signals = signals.copy()
    if active_start > 0:
        exec_signals.iloc[:active_start] = 0
    if active_end is not None:
        exec_signals.iloc[active_end:] = 0

    pnl, tv, n_long, n_short = execute_signals(
        job["te1"],
        job["te2"],
        exec_signals,
        beta_exec,
        cfg,
        allocation,
    )
    n_trades = sum(n_long.values()) + sum(n_short.values())
    metrics = compute_metrics(pnl / allocation, n_trades, tv / allocation)

    return {
        "signals": exec_signals,
        "raw_signals": signals,
        "beta_exec": beta_exec,
        "pnl": pnl,
        "tv": tv,
        "n_long_yr": n_long,
        "n_short_yr": n_short,
        "n_trades": n_trades,
        "metrics": metrics,
        "alpha_L": signal._alpha_L,
        "alpha_S": signal._alpha_S,
        "warmup_used": min(warmup_days, len(exec_signals)),
    }


def _select_mode_from_sweep(sweep_df: pd.DataFrame) -> str:
    informative = sweep_df[sweep_df["informative"]].copy()
    if informative.empty:
        mean_d = float(sweep_df["pnl_decile"].mean())
        mean_q = float(sweep_df["pnl_quintile"].mean())
        return "decile" if mean_d >= mean_q else "quintile"

    counts = informative["selected_mode"].value_counts()
    n_decile = int(counts.get("decile", 0))
    n_quintile = int(counts.get("quintile", 0))
    if n_decile != n_quintile:
        return "decile" if n_decile > n_quintile else "quintile"

    mean_d = float(informative["pnl_decile"].mean())
    mean_q = float(informative["pnl_quintile"].mean())
    return "decile" if mean_d >= mean_q else "quintile"


def _aggregate_runs(
    runs: list[dict],
    cfg: BacktestConfig,
    window_label: str,
    test_start: str,
    test_end: str,
    model_label: str,
    n_pairs_selected: int,
    n_pairs_tradable: int,
) -> dict:
    if not runs:
        return {}

    pnl_df = pd.DataFrame({run["pair"]: run["pnl"] for run in runs}).fillna(0.0)
    tv_df = pd.DataFrame({run["pair"]: run["tv"] for run in runs}).fillna(0.0)
    port_pnl = pnl_df.sum(axis=1)
    port_tv = tv_df.sum(axis=1)

    n_long_yr: dict[int, int] = {}
    n_short_yr: dict[int, int] = {}
    for run in runs:
        for year, count in run["n_long_yr"].items():
            n_long_yr[year] = n_long_yr.get(year, 0) + count
        for year, count in run["n_short_yr"].items():
            n_short_yr[year] = n_short_yr.get(year, 0) + count

    daily_ret = port_pnl / cfg.initial_capital
    daily_tv = port_tv / cfg.initial_capital
    n_trades = sum(n_long_yr.values()) + sum(n_short_yr.values())

    return {
        "window": window_label,
        "test_start": test_start,
        "test_end": test_end,
        "n_pairs": len(runs),
        "n_pairs_selected": n_pairs_selected,
        "n_pairs_tradable": n_pairs_tradable,
        "model": model_label,
        "daily_returns": daily_ret,
        "daily_turnover": daily_tv,
        "daily_pnl": port_pnl,
        "n_long_yr": n_long_yr,
        "n_short_yr": n_short_yr,
        "n_trades_total": n_trades,
        "metrics": compute_metrics(daily_ret, n_trades, daily_tv),
    }


def _evaluate_qualified_window(
    engine: BacktestEngine,
    predictions_root: Path,
    cfg: BacktestConfig,
    window_label: str,
    train_end: str,
    test_start: str,
    test_end: str,
    horizon: int,
    warmup_days: int,
    qualification_days: int,
    max_live_pairs: int,
    forced_mode: str | None,
    default_mode: str,
) -> tuple[dict, pd.DataFrame, int, int]:
    wide = engine._wide
    pairs = engine.pairs

    mask = (
        (pairs["training_window"] == window_label)
        & pairs["is_eligible"].astype(bool)
        & (pairs["score"] >= cfg.min_score)
    )
    window_pairs = (
        pairs[mask]
        .sort_values("score", ascending=False)
        .head(cfg.n_top_pairs)
    )
    if window_pairs.empty:
        return {}, pd.DataFrame(), 0, 0

    train_close = wide.loc[:train_end]
    test_close = wide.loc[test_start:test_end]
    if test_close.empty:
        return {}, pd.DataFrame(), len(window_pairs), 0

    coverage_signal = ForecastSignal(
        predictions_root,
        horizon=horizon,
        threshold_mode="decile",
        warmup_days=warmup_days,
    )

    tradable_jobs: list[dict] = []
    for _, row in window_pairs.iterrows():
        if not coverage_signal.has_pair_predictions(window_label, row["pair"]):
            continue
        job = _build_pair_job(row, train_close, test_close)
        if job is not None:
            tradable_jobs.append(job)

    if not tradable_jobs:
        return {}, pd.DataFrame(), len(window_pairs), 0

    if forced_mode is not None:
        candidate_modes = [forced_mode]
    elif qualification_days <= 0:
        candidate_modes = [default_mode]
    else:
        candidate_modes = ["decile", "quintile"]

    selection_allocation = 100_000.0
    selection_rows: list[dict] = []
    live_candidates: list[dict] = []

    for job in tradable_jobs:
        warmup_used = min(warmup_days, len(job["te1"]))
        qualification_end = min(len(job["te1"]), warmup_used + max(qualification_days, 0))

        best_row: dict | None = None

        for mode in candidate_modes:
            qual_run = _simulate_pair_mode(
                job,
                predictions_root,
                window_label,
                cfg,
                horizon,
                mode,
                warmup_days,
                selection_allocation,
                active_start=warmup_used,
                active_end=qualification_end if qualification_days > 0 else None,
            )

            row = {
                "pair": job["pair"],
                "mode": mode,
                "qual_pnl": float(qual_run["pnl"].sum()),
                "qual_trades": qual_run["n_trades"],
                "qual_return": qual_run["metrics"]["total_return"],
                "qual_sharpe": qual_run["metrics"]["sharpe"],
                "qual_fitness": qual_run["metrics"]["fitness"],
                "qual_turnover": qual_run["metrics"]["turnover"],
                "alpha_L": qual_run["alpha_L"],
                "alpha_S": qual_run["alpha_S"],
                "warmup_used": warmup_used,
                "qualification_end_idx": qualification_end,
            }

            better = False
            if best_row is None:
                better = True
            elif row["qual_fitness"] > best_row["qual_fitness"]:
                better = True
            elif row["qual_fitness"] == best_row["qual_fitness"] and row["qual_pnl"] > best_row["qual_pnl"]:
                better = True

            if better:
                best_row = row

            selection_rows.append(row)

        assert best_row is not None
        best_row["selected_mode"] = best_row["mode"]
        best_row["selected"] = (
            qualification_days <= 0
            or (
                best_row["qual_trades"] > 0
                and best_row["qual_pnl"] > 0.0
                and best_row["qual_fitness"] > 0.0
            )
        )

        if best_row["selected"]:
            live_candidates.append(
                {
                    "job": job,
                    "mode": best_row["selected_mode"],
                    "warmup_used": warmup_used,
                    "live_start": qualification_end if qualification_days > 0 else warmup_used,
                    "selection": best_row,
                }
            )

    live_candidates.sort(
        key=lambda item: (
            item["selection"]["qual_fitness"],
            item["selection"]["qual_pnl"],
        ),
        reverse=True,
    )
    if max_live_pairs > 0:
        live_candidates = live_candidates[:max_live_pairs]

    live_runs: list[dict] = []
    live_pairs = {item["job"]["pair"] for item in live_candidates}
    live_allocation = cfg.initial_capital / max(len(live_candidates), 1)

    for item in live_candidates:
        live_run = _simulate_pair_mode(
            item["job"],
            predictions_root,
            window_label,
            cfg,
            horizon,
            item["mode"],
            warmup_days,
            live_allocation,
            active_start=item["live_start"],
        )
        live_run["pair"] = item["job"]["pair"]
        live_run["selected_mode"] = item["mode"]
        live_run["live_start_idx"] = item["live_start"]
        live_runs.append(live_run)

    for row in selection_rows:
        row["selected_live"] = row["pair"] in live_pairs and row["mode"] == row.get("selected_mode")

    if live_runs:
        res = _aggregate_runs(
            live_runs,
            cfg,
            window_label,
            test_start,
            test_end,
            "ForecastSignalQualified",
            n_pairs_selected=len(window_pairs),
            n_pairs_tradable=len(tradable_jobs),
        )
    else:
        zero_idx = test_close.index
        zero_series = pd.Series(0.0, index=zero_idx)
        res = {
            "window": window_label,
            "test_start": test_start,
            "test_end": test_end,
            "n_pairs": 0,
            "n_pairs_selected": len(window_pairs),
            "n_pairs_tradable": len(tradable_jobs),
            "model": "ForecastSignalQualified",
            "daily_returns": zero_series.copy(),
            "daily_turnover": zero_series.copy(),
            "daily_pnl": zero_series.copy(),
            "n_long_yr": {},
            "n_short_yr": {},
            "n_trades_total": 0,
            "metrics": compute_metrics(zero_series, 0, zero_series),
        }

    return res, pd.DataFrame(selection_rows), len(window_pairs), len(tradable_jobs)


def tune_strategy_parameters(
    model_name: str,
    cfg: BacktestConfig,
    horizon: int,
    sweep_df: pd.DataFrame,
    warmup_candidates: list[int],
    qualification_candidates: list[int],
    max_live_pair_candidates: list[int],
) -> tuple[dict, pd.DataFrame]:
    predictions_root = PROJECT_ROOT / "data" / "processed" / "predictions" / model_name
    engine = BacktestEngine(cfg)
    engine.load_data()
    default_mode = _select_mode_from_sweep(sweep_df)

    rows: list[dict] = []
    for warmup_days in warmup_candidates:
        for qualification_days in qualification_candidates:
            for max_live_pairs in max_live_pair_candidates:
                fold_rows: list[dict] = []
                for fold in DEFAULT_CONFIG.expanding_folds:
                    res, _, _, _ = _evaluate_qualified_window(
                        engine=engine,
                        predictions_root=predictions_root,
                        cfg=cfg,
                        window_label=fold.label,
                        train_end=fold.train.end,
                        test_start=fold.val.start,
                        test_end=fold.val.end,
                        horizon=horizon,
                        warmup_days=warmup_days,
                        qualification_days=qualification_days,
                        max_live_pairs=max_live_pairs,
                        forced_mode=None,
                        default_mode=default_mode,
                    )
                    if not res:
                        continue
                    metrics = res["metrics"]
                    fold_rows.append(
                        {
                            "window": fold.label,
                            "total_return": metrics["total_return"],
                            "sharpe": metrics["sharpe"],
                            "fitness": metrics["fitness"],
                            "turnover": metrics["turnover"],
                            "n_trades": metrics["n_trades"],
                            "live_pairs": res["n_pairs"],
                        }
                    )

                if not fold_rows:
                    continue

                fold_df = pd.DataFrame(fold_rows)
                rows.append(
                    {
                        "warmup_days": warmup_days,
                        "qualification_days": qualification_days,
                        "max_live_pairs": max_live_pairs,
                        "n_folds": len(fold_df),
                        "mean_return": float(fold_df["total_return"].mean()),
                        "mean_sharpe": float(fold_df["sharpe"].mean()),
                        "mean_fitness": float(fold_df["fitness"].mean()),
                        "mean_turnover": float(fold_df["turnover"].mean()),
                        "mean_live_pairs": float(fold_df["live_pairs"].mean()),
                        "total_trades": int(fold_df["n_trades"].sum()),
                    }
                )

    tuning_df = pd.DataFrame(rows)
    if tuning_df.empty:
        fallback = {
            "warmup_days": warmup_candidates[0],
            "qualification_days": qualification_candidates[0],
            "max_live_pairs": max_live_pair_candidates[0],
        }
        return fallback, tuning_df

    tuning_df.sort_values(
        ["mean_return", "mean_sharpe", "mean_fitness", "mean_turnover"],
        ascending=[False, False, False, True],
        inplace=True,
    )
    tuning_df.reset_index(drop=True, inplace=True)

    out_dir = PROJECT_ROOT / "outputs" / "forecast_strategy" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    tuning_df.to_csv(out_dir / "strategy_tuning.csv", index=False)

    best = tuning_df.iloc[0]
    best_params = {
        "warmup_days": int(best["warmup_days"]),
        "qualification_days": int(best["qualification_days"]),
        "max_live_pairs": int(best["max_live_pairs"]),
    }
    return best_params, tuning_df


def run_validation_sweep(
    model_name: str,
    cfg: BacktestConfig,
    horizon: int = 10,
    warmup_days: int = 30,
) -> pd.DataFrame:
    predictions_root = PROJECT_ROOT / "data" / "processed" / "predictions" / model_name
    if not predictions_root.exists():
        print(f"[ERROR] Predictions not found: {predictions_root}")
        return pd.DataFrame()

    engine = BacktestEngine(cfg)
    engine.load_data()
    wide = engine._wide
    pairs = engine.pairs

    print(f"[ForecastSignal] Model: {model_name}")
    print(
        f"  Prices: {wide.shape[1]} tickers, "
        f"{wide.index.min().date()} – {wide.index.max().date()}"
    )
    print(f"  Pairs: {len(pairs)} total, {int(pairs['is_eligible'].sum())} eligible\n")

    results: list[dict] = []

    for fold in DEFAULT_CONFIG.expanding_folds:
        window_label = fold.label
        print(f"[{window_label}] Val {fold.val.start} → {fold.val.end}")

        mask = (
            (pairs["training_window"] == window_label)
            & pairs["is_eligible"].astype(bool)
            & (pairs["score"] >= cfg.min_score)
        )
        window_pairs = (
            pairs[mask]
            .sort_values("score", ascending=False)
            .head(cfg.n_top_pairs)
        )
        if window_pairs.empty:
            print("  [SKIP] No eligible pairs")
            continue

        train_close = wide.loc[:fold.train.end]
        val_close = wide.loc[fold.val.start:fold.val.end]
        if val_close.empty:
            print("  [SKIP] No validation prices")
            continue

        coverage_signal = ForecastSignal(
            predictions_root,
            horizon=horizon,
            threshold_mode="decile",
            warmup_days=warmup_days,
        )

        tradable_jobs: list[dict] = []
        for _, row in window_pairs.iterrows():
            if not coverage_signal.has_pair_predictions(window_label, row["pair"]):
                continue
            job = _build_pair_job(row, train_close, val_close)
            if job is not None:
                tradable_jobs.append(job)

        if not tradable_jobs:
            print("  [SKIP] No pairs with prediction coverage")
            continue

        allocation = cfg.initial_capital / len(tradable_jobs)

        for job in tradable_jobs:
            row_result = {
                "window": window_label,
                "pair": job["pair"],
            }

            for mode in ("decile", "quintile"):
                run = _simulate_pair_mode(
                    job,
                    predictions_root,
                    window_label,
                    cfg,
                    horizon,
                    mode,
                    warmup_days,
                    allocation,
                )
                row_result[f"pnl_{mode}"] = float(run["pnl"].sum())
                row_result[f"alpha_L_{mode}"] = run["alpha_L"]
                row_result[f"alpha_S_{mode}"] = run["alpha_S"]
                row_result[f"n_trades_{mode}"] = run["n_trades"]
                row_result[f"fitness_{mode}"] = run["metrics"]["fitness"]
                row_result[f"turnover_{mode}"] = run["metrics"]["turnover"]

            informative = (
                row_result["n_trades_decile"] > 0 or row_result["n_trades_quintile"] > 0
            )
            row_result["informative"] = informative

            if not informative:
                row_result["selected_mode"] = "inactive"
            elif row_result["n_trades_decile"] == 0:
                row_result["selected_mode"] = "quintile"
            elif row_result["n_trades_quintile"] == 0:
                row_result["selected_mode"] = "decile"
            elif row_result["fitness_decile"] > row_result["fitness_quintile"]:
                row_result["selected_mode"] = "decile"
            elif row_result["fitness_quintile"] > row_result["fitness_decile"]:
                row_result["selected_mode"] = "quintile"
            elif row_result["pnl_decile"] >= row_result["pnl_quintile"]:
                row_result["selected_mode"] = "decile"
            else:
                row_result["selected_mode"] = "quintile"

            results.append(row_result)

        n_done = sum(1 for r in results if r["window"] == window_label)
        print(f"  → Evaluated {n_done} pairs with prediction coverage")

    df = pd.DataFrame(results)
    if df.empty:
        print("\n[WARN] No pairs evaluated during validation sweep")
        return df

    informative = df[df["informative"]].copy()
    n_informative = len(informative)
    n_decile = int((informative["selected_mode"] == "decile").sum())
    n_quintile = int((informative["selected_mode"] == "quintile").sum())

    print(f"\n{'═' * 60}")
    print("  VALIDATION SWEEP SUMMARY")
    print(f"{'═' * 60}")
    print(f"  Total pair-windows evaluated: {len(df)}")
    print(f"  Informative rows:            {n_informative}")
    if n_informative > 0:
        print(f"  Decile wins:                 {n_decile} ({100 * n_decile / n_informative:.1f}%)")
        print(f"  Quintile wins:               {n_quintile} ({100 * n_quintile / n_informative:.1f}%)")
        print(f"  Avg P&L decile (active):     ${informative['pnl_decile'].mean():,.2f}")
        print(f"  Avg P&L quintile (active):   ${informative['pnl_quintile'].mean():,.2f}")
    else:
        print("  No informative rows; both modes were inactive throughout.")
    print(f"  Zero-trade decile rows:      {int((df['n_trades_decile'] == 0).sum())}")
    print(f"  Zero-trade quintile rows:    {int((df['n_trades_quintile'] == 0).sum())}")
    print(f"{'═' * 60}\n")

    out_dir = PROJECT_ROOT / "outputs" / "forecast_strategy" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "validation_sweep.csv"
    df.to_csv(out_path, index=False)
    print(f"  Saved: {out_path}\n")

    return df


def run_holdout(
    model_name: str,
    cfg: BacktestConfig,
    horizon: int = 10,
    warmup_days: int = 30,
    qualification_days: int = 30,
    max_live_pairs: int = 5,
    forced_mode: str | None = None,
    default_mode: str = "decile",
    sweep_df: pd.DataFrame | None = None,
    plot: bool = True,
) -> dict:
    predictions_root = PROJECT_ROOT / "data" / "processed" / "predictions" / model_name
    holdout = DEFAULT_CONFIG.holdout_split
    holdout_path = predictions_root / holdout.label / "predictions.csv"
    if not holdout_path.exists():
        print(f"[ERROR] Holdout predictions not found: {holdout_path}")
        return {}

    print(f"\n{'═' * 60}")
    print(f"  HOLDOUT: {model_name}")
    print(f"{'═' * 60}")
    print(f"  Warmup days:       {warmup_days}")
    print(f"  Qualification days:{qualification_days}")
    if forced_mode is not None:
        print(f"  Mode policy:       forced {forced_mode}")
    else:
        print("  Mode policy:       pair-specific qualification")
    print(f"  Max live pairs:    {max_live_pairs}\n")

    engine = BacktestEngine(cfg)
    engine.load_data()

    if qualification_days <= 0:
        print(f"  Validation default mode: {default_mode}")
    elif sweep_df is not None and not sweep_df.empty:
        print(f"  Validation default mode: {default_mode}")
    else:
        print("  Validation default mode: decile (fallback)")

    res, selection_df, n_pairs_selected, n_pairs_tradable = _evaluate_qualified_window(
        engine=engine,
        predictions_root=predictions_root,
        cfg=cfg,
        window_label=holdout.label,
        train_end=holdout.train.end,
        test_start=holdout.test.start,
        test_end=holdout.test.end,
        horizon=horizon,
        warmup_days=warmup_days,
        qualification_days=qualification_days,
        max_live_pairs=max_live_pairs,
        forced_mode=forced_mode,
        default_mode=default_mode,
    )
    if not res:
        print("[ERROR] Holdout evaluation produced no results.")
        return {}

    results = {holdout.label: res, "__aggregate__": res}

    live_pairs = int(selection_df["selected_live"].sum()) if not selection_df.empty else 0
    print(
        f"[{holdout.label}]  selected={n_pairs_selected:3d}  |  "
        f"tradable={n_pairs_tradable:3d}  |  live={live_pairs:3d}  |  "
        f"test {holdout.test.start} → {holdout.test.end}  ({len(engine._wide.loc[holdout.test.start:holdout.test.end])} days)"
    )

    engine.report(results)
    if plot:
        try:
            engine.plot(results)
        except Exception as exc:
            print(f"  [WARN] Plot failed: {exc}")

    out_dir = Path(cfg.output_dir) / "forecast" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)

    if not selection_df.empty:
        selection_df.sort_values(
            ["selected_live", "qual_fitness", "qual_pnl"],
            ascending=[False, False, False],
            inplace=True,
        )
        selection_path = out_dir / "holdout_pair_selection.csv"
        selection_df.to_csv(selection_path, index=False)
        print(f"  Saved pair selection: {selection_path}")

    metrics_df = pd.DataFrame([res["metrics"]])
    metrics_df["model"] = model_name
    metrics_df["warmup_days"] = warmup_days
    metrics_df["qualification_days"] = qualification_days
    metrics_df["max_live_pairs"] = max_live_pairs
    metrics_df["live_pairs"] = live_pairs
    metrics_df.to_csv(out_dir / "holdout_metrics.csv", index=False)
    print(f"  Saved metrics: {out_dir / 'holdout_metrics.csv'}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run forecast signal strategy with causal calibration"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="xgboost_ols",
        help="Model name matching data/processed/predictions/<model>",
    )
    parser.add_argument(
        "--holdout",
        action="store_true",
        help="Run holdout after validation sweep",
    )
    parser.add_argument(
        "--mode",
        type=str,
        default=None,
        choices=["decile", "quintile"],
        help="Force one threshold mode instead of pair-specific mode selection",
    )
    parser.add_argument(
        "--horizon",
        type=int,
        default=10,
        help="Forecast horizon in trading days",
    )
    parser.add_argument(
        "--warmup_days",
        type=int,
        default=30,
        help="Causal warmup days used to calibrate threshold levels",
    )
    parser.add_argument(
        "--qualification_days",
        type=int,
        default=30,
        help="Holdout qualification days used to select live pairs and modes",
    )
    parser.add_argument(
        "--max_live_pairs",
        type=int,
        default=5,
        help="Maximum pairs to trade after qualification (0 = no cap)",
    )
    parser.add_argument(
        "--n_pairs",
        type=int,
        default=50,
        help="Max candidate pairs per window",
    )
    parser.add_argument(
        "--no_plot",
        action="store_true",
        help="Suppress plots",
    )
    parser.add_argument(
        "--no_strategy_tune",
        action="store_true",
        help="Skip validation-based tuning of warmup/qualification/live-pair parameters",
    )
    args = parser.parse_args()

    cfg = BacktestConfig(n_top_pairs=args.n_pairs)

    sweep_df = run_validation_sweep(
        args.model,
        cfg,
        horizon=args.horizon,
        warmup_days=args.warmup_days,
    )
    if sweep_df.empty:
        print("[ERROR] Validation sweep produced no results. Exiting.")
        sys.exit(1)

    selected_mode = _select_mode_from_sweep(sweep_df)
    print(f"  → Validation default mode: {selected_mode}\n")

    if args.holdout:
        warmup_days = args.warmup_days
        qualification_days = args.qualification_days
        max_live_pairs = args.max_live_pairs

        if not args.no_strategy_tune:
            warmup_grid = [args.warmup_days] if args.warmup_days != 30 else [20, 30]
            qualification_grid = (
                [args.qualification_days]
                if args.qualification_days != 30
                else [20, 30, 40]
            )
            max_live_pair_grid = (
                [args.max_live_pairs]
                if args.max_live_pairs != 5
                else [3, 5]
            )

            best_params, tuning_df = tune_strategy_parameters(
                args.model,
                cfg,
                horizon=args.horizon,
                sweep_df=sweep_df,
                warmup_candidates=warmup_grid,
                qualification_candidates=qualification_grid,
                max_live_pair_candidates=max_live_pair_grid,
            )

            if not tuning_df.empty:
                print(f"{'═' * 60}")
                print("  STRATEGY TUNING")
                print(f"{'═' * 60}")
                best_mean_return = float(tuning_df.iloc[0]["mean_return"])
                best_mean_sharpe = float(tuning_df.iloc[0]["mean_sharpe"])
                use_tuned = best_mean_return > 0.0 and best_mean_sharpe > 0.0
                if use_tuned:
                    print(
                        f"  Selected warmup={best_params['warmup_days']}  "
                        f"qualification={best_params['qualification_days']}  "
                        f"max_live_pairs={best_params['max_live_pairs']}"
                    )
                else:
                    print(
                        "  Validation tuning did not show positive trading edge; "
                        "keeping requested live parameters."
                    )
                print(
                    f"  Best validation mean return: "
                    f"{best_mean_return:.2%}"
                )
                print(
                    f"  Best validation mean Sharpe: "
                    f"{best_mean_sharpe:.2f}"
                )
                print(
                    f"  Saved: "
                    f"{PROJECT_ROOT / 'outputs' / 'forecast_strategy' / args.model / 'strategy_tuning.csv'}"
                )
                print(f"{'═' * 60}\n")
                if use_tuned:
                    warmup_days = best_params["warmup_days"]
                    qualification_days = best_params["qualification_days"]
                    max_live_pairs = best_params["max_live_pairs"]

        run_holdout(
            args.model,
            cfg,
            horizon=args.horizon,
            warmup_days=warmup_days,
            qualification_days=qualification_days,
            max_live_pairs=max_live_pairs,
            forced_mode=args.mode,
            default_mode=selected_mode,
            sweep_df=sweep_df,
            plot=not args.no_plot,
        )


if __name__ == "__main__":
    main()
