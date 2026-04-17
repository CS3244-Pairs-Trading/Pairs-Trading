#!/usr/bin/env python3
"""
Trading-aware final model selection for forecast models.

Workflow:
1. Aggregate validation forecast metrics from saved prediction artifacts.
2. Shortlist the top-N model variants by validation forecast error.
3. Run the forecast strategy on the validation folds and rank the shortlist
   by validation trading Sharpe / Fitness.
4. Optionally run the selected validation strategy on holdout where
   predictions exist, but keep holdout strictly as a report card.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Iterable

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

from scripts.run_forecast_signal import (
    _evaluate_qualified_window,
    _select_mode_from_sweep,
    run_holdout,
    run_validation_sweep,
    tune_strategy_parameters,
)
from src.backtest.backtest_engine import BacktestConfig, BacktestEngine
from src.config import DEFAULT_CONFIG
from src.models.prediction_metrics import (
    DEFAULT_DIRECTIONAL_MSE_GAMMA,
    evaluate_regression_predictions,
)


DEFAULT_WARMUP = 30
DEFAULT_QUALIFICATION = 30
DEFAULT_MAX_LIVE_PAIRS = 5
DEFAULT_TOP_N_FORECAST = 3


def _parse_model_variant(model_name: str) -> tuple[str, str]:
    name = model_name.lower()
    if name.endswith("_kalman"):
        return model_name[: -len("_kalman")], "kalman"
    if name.endswith("_ols"):
        return model_name[: -len("_ols")], "ols"
    if name.endswith("_static"):
        return model_name[: -len("_static")], "ols"
    return model_name, "kalman" if "kalman" in name else "ols"


def _display_model_family(model_name: str) -> str:
    base, _ = _parse_model_variant(model_name)
    aliases = {
        "linear_regression": "linear",
        "lstm_encoder_decoder": "lstm_encdec",
    }
    return aliases.get(base, base)


def _predictions_root(model_name: str) -> Path:
    return PROJECT_ROOT / "data" / "processed" / "predictions" / model_name


def _target_col(model_name: str) -> str:
    _, spread_type = _parse_model_variant(model_name)
    return "label_kalman_10d" if spread_type == "kalman" else "label_continuous_10d"


def _prediction_col(df: pd.DataFrame) -> str:
    if "predicted_change" in df.columns:
        return "predicted_change"
    if "predicted_spread_change" in df.columns:
        return "predicted_spread_change"
    raise ValueError("Prediction file missing 'predicted_change' / 'predicted_spread_change'.")


def _compute_forecast_metrics(
    actual: np.ndarray,
    predicted: np.ndarray,
    directional_mse_gamma: float = DEFAULT_DIRECTIONAL_MSE_GAMMA,
) -> dict[str, float]:
    return evaluate_regression_predictions(
        actual,
        predicted,
        gamma=directional_mse_gamma,
    )


def _load_matched_window_predictions(
    model_name: str,
    window_label: str,
    split_name: str,
) -> pd.DataFrame:
    pred_path = _predictions_root(model_name) / window_label / "predictions.csv"
    data_path = (
        PROJECT_ROOT / "data" / "processed" / "pair_datasets" / window_label / f"{split_name}_pair_dataset.csv"
    )
    if not pred_path.exists() or not data_path.exists():
        return pd.DataFrame()

    pred_df = pd.read_csv(pred_path, parse_dates=["Date"])
    data_df = pd.read_csv(data_path, parse_dates=["Date"])
    pred_col = _prediction_col(pred_df)
    target_col = _target_col(model_name)

    pred_df = pred_df[["Date", "pair", pred_col]].copy()
    pred_df.rename(columns={pred_col: "predicted_change"}, inplace=True)
    pred_df.dropna(subset=["Date", "pair", "predicted_change"], inplace=True)

    data_df = data_df[["Date", "pair", target_col]].copy()
    data_df.rename(columns={target_col: "actual_change"}, inplace=True)
    data_df.dropna(subset=["Date", "pair", "actual_change"], inplace=True)

    merged = pred_df.merge(data_df, on=["Date", "pair"], how="inner")
    return merged.dropna(subset=["predicted_change", "actual_change"]).reset_index(drop=True)


def _aggregate_metric_rows(
    rows: list[dict],
    prefix: str,
) -> dict[str, float]:
    if not rows:
        return {
            f"{prefix}_n_windows": 0,
            f"{prefix}_n_rows": 0,
        }

    df = pd.DataFrame(rows)
    out: dict[str, float] = {
        f"{prefix}_n_windows": int(df["window"].nunique()),
        f"{prefix}_n_rows": int(df["n_rows"].sum()),
    }
    for metric in [
        "directional_weighted_mse",
        "rmse",
        "r2",
        "information_coefficient",
        "directional_accuracy",
        "profit_weighted_da",
    ]:
        out[f"{prefix}_{metric}_mean"] = float(df[metric].mean())
        out[f"{prefix}_{metric}_std"] = float(df[metric].std(ddof=0))
    return out


def aggregate_forecast_summary(
    model_name: str,
    directional_mse_gamma: float = DEFAULT_DIRECTIONAL_MSE_GAMMA,
) -> dict[str, float]:
    base_model, spread_type = _parse_model_variant(model_name)
    validation_rows: list[dict] = []

    for fold in DEFAULT_CONFIG.expanding_folds:
        merged = _load_matched_window_predictions(model_name, fold.label, "val")
        if merged.empty:
            continue
        metrics = _compute_forecast_metrics(
            merged["actual_change"].to_numpy(dtype=float),
            merged["predicted_change"].to_numpy(dtype=float),
            directional_mse_gamma=directional_mse_gamma,
        )
        validation_rows.append({
            "window": fold.label,
            "n_rows": len(merged),
            **metrics,
        })

    summary = {
        "model_name": model_name,
        "model": _display_model_family(model_name),
        "spread_type": spread_type,
        "directional_mse_gamma": float(directional_mse_gamma),
    }
    summary.update(_aggregate_metric_rows(validation_rows, "validation"))

    holdout_label = DEFAULT_CONFIG.holdout_split.label
    holdout_merged = _load_matched_window_predictions(model_name, holdout_label, "test")
    if holdout_merged.empty:
        summary["holdout_n_windows"] = 0
        summary["holdout_n_rows"] = 0
    else:
        holdout_metrics = _compute_forecast_metrics(
            holdout_merged["actual_change"].to_numpy(dtype=float),
            holdout_merged["predicted_change"].to_numpy(dtype=float),
            directional_mse_gamma=directional_mse_gamma,
        )
        summary.update({
            "holdout_n_windows": 1,
            "holdout_n_rows": int(len(holdout_merged)),
            **{f"holdout_{k}": float(v) for k, v in holdout_metrics.items()},
        })
    return summary


def _default_strategy_grids(
    warmup_days: int,
    qualification_days: int,
    max_live_pairs: int,
) -> tuple[list[int], list[int], list[int]]:
    warmup_grid = [warmup_days] if warmup_days != DEFAULT_WARMUP else [20, 30]
    qualification_grid = (
        [qualification_days] if qualification_days != DEFAULT_QUALIFICATION else [20, 30, 40]
    )
    live_pair_grid = [max_live_pairs] if max_live_pairs != DEFAULT_MAX_LIVE_PAIRS else [3, 5]
    return warmup_grid, qualification_grid, live_pair_grid


def evaluate_validation_trading(
    model_name: str,
    cfg: BacktestConfig,
    horizon: int,
    warmup_days: int,
    qualification_days: int,
    max_live_pairs: int,
    strategy_tune: bool,
) -> tuple[dict[str, float], pd.DataFrame, pd.DataFrame]:
    sweep_df = run_validation_sweep(
        model_name=model_name,
        cfg=cfg,
        horizon=horizon,
        warmup_days=warmup_days,
        score_mode="auto",
        entry_scale=1.0,
        reentry_cooldown_days=0,
    )
    if sweep_df.empty:
        return {}, pd.DataFrame(), pd.DataFrame()

    selected_mode = _select_mode_from_sweep(sweep_df)
    chosen_params = {
        "warmup_days": warmup_days,
        "qualification_days": qualification_days,
        "max_live_pairs": max_live_pairs,
    }
    tuning_df = pd.DataFrame()
    used_strategy_tune = False

    if strategy_tune:
        warmup_grid, qualification_grid, live_pair_grid = _default_strategy_grids(
            warmup_days,
            qualification_days,
            max_live_pairs,
        )
        best_params, tuning_df = tune_strategy_parameters(
            model_name=model_name,
            cfg=cfg,
            horizon=horizon,
            sweep_df=sweep_df,
            warmup_candidates=warmup_grid,
            qualification_candidates=qualification_grid,
            max_live_pair_candidates=live_pair_grid,
            score_mode="auto",
            entry_scale_candidates=[1.0],
            reentry_cooldown_candidates=[0],
        )
        if not tuning_df.empty:
            best_mean_return = float(tuning_df.iloc[0]["mean_return"])
            best_mean_sharpe = float(tuning_df.iloc[0]["mean_sharpe"])
            best_recent_sharpe = float(tuning_df.iloc[0].get("recent_sharpe", best_mean_sharpe))
            if best_mean_return > 0.0 and best_mean_sharpe > 0.0 and best_recent_sharpe > 0.0:
                chosen_params = best_params
                used_strategy_tune = True

    engine = BacktestEngine(cfg)
    engine.load_data()
    predictions_root = _predictions_root(model_name)

    fold_rows: list[dict] = []
    for fold in DEFAULT_CONFIG.expanding_folds:
        res, selection_df, _, _ = _evaluate_qualified_window(
            engine=engine,
            predictions_root=predictions_root,
            cfg=cfg,
            window_label=fold.label,
            train_end=fold.train.end,
            test_start=fold.val.start,
            test_end=fold.val.end,
            horizon=horizon,
            warmup_days=int(chosen_params["warmup_days"]),
            qualification_days=int(chosen_params["qualification_days"]),
            max_live_pairs=int(chosen_params["max_live_pairs"]),
            score_mode="auto",
            entry_scale=float(chosen_params.get("entry_scale", 1.0)),
            reentry_cooldown_days=int(chosen_params.get("reentry_cooldown_days", 0)),
            forced_mode=None,
            default_mode=selected_mode,
        )
        if not res:
            continue

        metrics = res["metrics"]
        live_pairs = (
            int(selection_df["selected_live"].sum())
            if not selection_df.empty and "selected_live" in selection_df.columns
            else int(res["n_pairs"])
        )
        fold_rows.append({
            "window": fold.label,
            "total_return": metrics["total_return"],
            "annualized_return": metrics["annualized_return"],
            "sharpe": metrics["sharpe"],
            "fitness": metrics["fitness"],
            "turnover": metrics["turnover"],
            "max_drawdown": metrics["max_drawdown"],
            "n_trades": metrics["n_trades"],
            "live_pairs": live_pairs,
        })

    if not fold_rows:
        return {}, sweep_df, tuning_df

    fold_df = pd.DataFrame(fold_rows)
    summary = {
        "validation_trading_n_folds": int(len(fold_df)),
        "validation_trading_mean_return": float(fold_df["total_return"].mean()),
        "validation_trading_std_return": float(fold_df["total_return"].std(ddof=0)),
        "validation_trading_mean_annualized_return": float(fold_df["annualized_return"].mean()),
        "validation_trading_mean_sharpe": float(fold_df["sharpe"].mean()),
        "validation_trading_std_sharpe": float(fold_df["sharpe"].std(ddof=0)),
        "validation_trading_mean_fitness": float(fold_df["fitness"].mean()),
        "validation_trading_mean_turnover": float(fold_df["turnover"].mean()),
        "validation_trading_mean_drawdown": float(fold_df["max_drawdown"].mean()),
        "validation_trading_total_trades": int(fold_df["n_trades"].sum()),
        "validation_trading_mean_live_pairs": float(fold_df["live_pairs"].mean()),
        "strategy_default_mode": selected_mode,
        "strategy_tuned": bool(used_strategy_tune),
        "strategy_warmup_days": int(chosen_params["warmup_days"]),
        "strategy_qualification_days": int(chosen_params["qualification_days"]),
        "strategy_max_live_pairs": int(chosen_params["max_live_pairs"]),
        "strategy_score_mode": "auto",
        "strategy_entry_scale": float(chosen_params.get("entry_scale", 1.0)),
        "strategy_reentry_cooldown_days": int(chosen_params.get("reentry_cooldown_days", 0)),
    }
    return summary, sweep_df, tuning_df


def evaluate_holdout_trading(
    model_name: str,
    cfg: BacktestConfig,
    horizon: int,
    warmup_days: int,
    qualification_days: int,
    max_live_pairs: int,
    default_mode: str,
    sweep_df: pd.DataFrame,
) -> dict[str, float]:
    results = run_holdout(
        model_name=model_name,
        cfg=cfg,
        horizon=horizon,
        warmup_days=warmup_days,
        qualification_days=qualification_days,
        max_live_pairs=max_live_pairs,
        score_mode="auto",
        entry_scale=1.0,
        reentry_cooldown_days=0,
        forced_mode=None,
        default_mode=default_mode,
        sweep_df=sweep_df,
        plot=False,
    )
    if not results or "__aggregate__" not in results:
        return {}

    agg = results["__aggregate__"]
    metrics = agg["metrics"]
    return {
        "holdout_trading_total_return": float(metrics["total_return"]),
        "holdout_trading_annualized_return": float(metrics["annualized_return"]),
        "holdout_trading_sharpe": float(metrics["sharpe"]),
        "holdout_trading_fitness": float(metrics["fitness"]),
        "holdout_trading_turnover": float(metrics["turnover"]),
        "holdout_trading_max_drawdown": float(metrics["max_drawdown"]),
        "holdout_trading_n_trades": int(metrics["n_trades"]),
        "holdout_trading_live_pairs": int(agg["n_pairs"]),
    }


def discover_models(requested_models: Iterable[str] | None) -> list[str]:
    if requested_models:
        return list(dict.fromkeys(requested_models))

    root = PROJECT_ROOT / "data" / "processed" / "predictions"
    discovered: list[str] = []
    for path in sorted(p for p in root.iterdir() if p.is_dir()):
        has_validation_preds = any(
            (path / fold.label / "predictions.csv").exists() for fold in DEFAULT_CONFIG.expanding_folds
        )
        if has_validation_preds:
            discovered.append(path.name)
    return discovered


def build_model_selection_summary(
    models: list[str],
    cfg: BacktestConfig,
    horizon: int,
    warmup_days: int,
    qualification_days: int,
    max_live_pairs: int,
    top_n_forecast: int,
    strategy_tune: bool,
    run_holdout_eval: bool,
    directional_mse_gamma: float,
) -> pd.DataFrame:
    rows: list[dict] = []

    for model_name in models:
        print(f"\n{'=' * 72}")
        print(f"MODEL SELECTION — {model_name}")
        print(f"{'=' * 72}")

        row = aggregate_forecast_summary(
            model_name,
            directional_mse_gamma=directional_mse_gamma,
        )
        trading_summary, sweep_df, _ = evaluate_validation_trading(
            model_name=model_name,
            cfg=cfg,
            horizon=horizon,
            warmup_days=warmup_days,
            qualification_days=qualification_days,
            max_live_pairs=max_live_pairs,
            strategy_tune=strategy_tune,
        )
        if not trading_summary:
            print("  [WARN] No validation trading summary produced; skipping model.")
            continue

        row.update(trading_summary)

        if run_holdout_eval:
            holdout_summary = evaluate_holdout_trading(
                model_name=model_name,
                cfg=cfg,
                horizon=horizon,
                warmup_days=int(trading_summary["strategy_warmup_days"]),
                qualification_days=int(trading_summary["strategy_qualification_days"]),
                max_live_pairs=int(trading_summary["strategy_max_live_pairs"]),
                default_mode=str(trading_summary["strategy_default_mode"]),
                sweep_df=sweep_df,
            )
            row.update(holdout_summary)

        rows.append(row)

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    df.sort_values(
        [
            "validation_directional_weighted_mse_mean",
            "validation_rmse_mean",
            "validation_r2_mean",
            "validation_information_coefficient_mean",
        ],
        ascending=[True, True, False, False],
        inplace=True,
        na_position="last",
    )
    df.reset_index(drop=True, inplace=True)
    df["forecast_rank"] = np.arange(1, len(df) + 1)
    df["forecast_shortlisted"] = df["forecast_rank"] <= int(top_n_forecast)

    shortlist = df[df["forecast_shortlisted"]].copy()
    shortlist.sort_values(
        [
            "validation_trading_mean_sharpe",
            "validation_trading_mean_fitness",
            "validation_trading_mean_return",
            "validation_trading_mean_turnover",
        ],
        ascending=[False, False, False, True],
        inplace=True,
        na_position="last",
    )
    shortlist["trading_rank_within_shortlist"] = np.arange(1, len(shortlist) + 1)

    df = df.merge(
        shortlist[["model_name", "trading_rank_within_shortlist"]],
        on="model_name",
        how="left",
    )
    df["selected_for_live"] = df["trading_rank_within_shortlist"].fillna(np.inf).eq(1)
    return df


def print_selection_summary(df: pd.DataFrame) -> None:
    if df.empty:
        print("[ERROR] No model-selection rows produced.")
        return

    shortlist = df[df["forecast_shortlisted"]].copy()
    print(f"\n{'═' * 72}")
    print("  FINAL MODEL SELECTION SUMMARY")
    print(f"{'═' * 72}")
    gamma = df["directional_mse_gamma"].iloc[0] if "directional_mse_gamma" in df.columns else DEFAULT_DIRECTIONAL_MSE_GAMMA
    print(
        "  Shortlist rule: top models by validation directional-weighted MSE "
        f"(gamma={float(gamma):.2f}), final choice by validation trading Sharpe/Fitness."
    )
    print()
    cols = [
        "model_name",
        "forecast_rank",
        "directional_mse_gamma",
        "validation_directional_weighted_mse_mean",
        "validation_rmse_mean",
        "validation_r2_mean",
        "validation_information_coefficient_mean",
        "validation_directional_accuracy_mean",
        "validation_trading_mean_sharpe",
        "validation_trading_mean_fitness",
        "validation_trading_mean_return",
        "trading_rank_within_shortlist",
        "selected_for_live",
    ]
    present_cols = [c for c in cols if c in df.columns]
    print(df[present_cols].to_string(index=False, justify="left"))

    if not shortlist.empty:
        selected = shortlist.sort_values("trading_rank_within_shortlist").iloc[0]
        print(f"\n  Selected live model: {selected['model_name']}")
        print(
            f"  Validation Sharpe={selected['validation_trading_mean_sharpe']:.2f}  "
            f"Fitness={selected['validation_trading_mean_fitness']:.2f}  "
            f"Return={selected['validation_trading_mean_return']:.2%}"
        )
        if "holdout_trading_sharpe" in selected and pd.notna(selected["holdout_trading_sharpe"]):
            print(
                f"  Holdout Sharpe={selected['holdout_trading_sharpe']:.2f}  "
                f"Return={selected['holdout_trading_total_return']:.2%}"
            )
    print(f"{'═' * 72}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select the final live model by validation trading performance."
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Prediction folders under data/processed/predictions/. Default: auto-discover.",
    )
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--warmup_days", type=int, default=DEFAULT_WARMUP)
    parser.add_argument("--qualification_days", type=int, default=DEFAULT_QUALIFICATION)
    parser.add_argument("--max_live_pairs", type=int, default=DEFAULT_MAX_LIVE_PAIRS)
    parser.add_argument("--n_pairs", type=int, default=50)
    parser.add_argument("--top_n_forecast", type=int, default=DEFAULT_TOP_N_FORECAST)
    parser.add_argument(
        "--directional_mse_gamma",
        type=float,
        default=DEFAULT_DIRECTIONAL_MSE_GAMMA,
        help="Gamma penalty used in directional-weighted MSE.",
    )
    parser.add_argument(
        "--no_strategy_tune",
        action="store_true",
        help="Skip validation-based tuning of strategy warmup / qualification / live-pair parameters.",
    )
    parser.add_argument(
        "--skip_holdout",
        action="store_true",
        help="Do not run holdout trading evaluation for models with holdout predictions.",
    )
    args = parser.parse_args()

    models = discover_models(args.models)
    if not models:
        print("[ERROR] No models discovered under data/processed/predictions/")
        sys.exit(1)

    cfg = BacktestConfig(n_top_pairs=args.n_pairs)
    summary_df = build_model_selection_summary(
        models=models,
        cfg=cfg,
        horizon=args.horizon,
        warmup_days=args.warmup_days,
        qualification_days=args.qualification_days,
        max_live_pairs=args.max_live_pairs,
        top_n_forecast=args.top_n_forecast,
        strategy_tune=not args.no_strategy_tune,
        run_holdout_eval=not args.skip_holdout,
        directional_mse_gamma=args.directional_mse_gamma,
    )
    if summary_df.empty:
        print("[ERROR] Model selection failed to produce any summary rows.")
        sys.exit(1)

    out_dir = PROJECT_ROOT / "outputs" / "model_selection"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "model_selection_summary.csv"
    summary_df.to_csv(out_path, index=False)
    print(f"\nSaved summary: {out_path}")
    print_selection_summary(summary_df)


if __name__ == "__main__":
    main()
