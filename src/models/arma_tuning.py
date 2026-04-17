from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import DEFAULT_CONFIG
from src.models.arma import (
    iter_window_dirs,
    load_window_split_datasets,
    resolve_volatility_column,
    spread_variant_tag,
)

P_VALUES = [0,1,2,4,6,8,9,10]
Q_VALUES = [0,1,2,4,6,8,9,10]


def _parse_int_list(raw: str | None, default_values: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return default_values
    out = [int(x.strip()) for x in raw.split(",") if x.strip() != ""]
    if not out:
        return default_values
    return out


def _select_target_pairs(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pair: str | None = None,
) -> list[str]:
    train_pairs = set(train_df["pair"].dropna().astype(str).unique())
    eval_pairs = set(eval_df["pair"].dropna().astype(str).unique())
    shared = sorted(train_pairs.intersection(eval_pairs))

    if pair is None:
        return shared
    if pair not in shared:
        return []
    return [pair]


def _extract_pair_frame(
    df: pd.DataFrame,
    pair: str,
    spread_col: str,
    vol_col: str,
) -> pd.DataFrame:
    cols = ["Date", "pair", spread_col, vol_col]
    missing = set(cols) - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns: {sorted(missing)}")

    out = df.loc[df["pair"].astype(str) == str(pair), ["Date", spread_col, vol_col]].copy()
    if out.empty:
        return out

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[spread_col] = pd.to_numeric(out[spread_col], errors="coerce")
    out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce")

    return out.dropna(subset=["Date", spread_col]).sort_values("Date").reset_index(drop=True)


def _evaluate_fit_once_forecasts(forecast_df: pd.DataFrame) -> dict[str, float]:
    actual_change = forecast_df["actual_change"].astype(float).to_numpy()
    predicted_change = forecast_df["predicted_change"].astype(float).to_numpy()
    actual_value = forecast_df["actual_value"].astype(float).to_numpy()
    predicted_value = forecast_df["predicted_value"].astype(float).to_numpy()

    return {
        "mse": float(np.mean((actual_change - predicted_change) ** 2)),
        "mae": float(np.mean(np.abs(actual_change - predicted_change))),
        "mse_level": float(np.mean((actual_value - predicted_value) ** 2)),
        "mae_level": float(np.mean(np.abs(actual_value - predicted_value))),
    }


def fit_once_validate_pair(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    pair: str,
    spread_col: str,
    p: int,
    q: int,
    horizon: int,
    min_train_points: int,
    min_eval_points: int,
    window_label: str,
) -> tuple[pd.DataFrame, dict[str, float | int | str]] | None:
    """
    Cheap tuning-time proxy:
    - fit ARIMA(order=(p,0,q)) once on train spread
    - forecast full validation block once
    - derive horizon changes from forecasted future level
    """

    if horizon <= 0:
        return None

    vol_col = resolve_volatility_column(pd.concat([train_df, val_df], ignore_index=True), spread_col)

    train_pair_df = _extract_pair_frame(train_df, pair=pair, spread_col=spread_col, vol_col=vol_col)
    val_pair_df = _extract_pair_frame(val_df, pair=pair, spread_col=spread_col, vol_col=vol_col)

    n_train = len(train_pair_df)
    n_eval_points = len(val_pair_df)
    if n_train < min_train_points:
        return None
    if n_eval_points < max(min_eval_points, horizon + 1):
        return None
    
    p1 = val_pair_df["log_price_a"].astype(float).to_numpy()
    p2 = val_pair_df["log_price_b"].astype(float).to_numpy()
    betas = val_pair_df["kalman_beta"].astype(float).to_numpy()

    train_series = train_pair_df[spread_col].astype(float)
    eval_spread = val_pair_df[spread_col].astype(float).to_numpy()
    eval_dates = pd.to_datetime(val_pair_df["Date"])
    eval_vol = val_pair_df[vol_col].astype(float).to_numpy()

    try:
        fitted = ARIMA(train_series, order=(int(p), 0, int(q))).fit()
        forecast_values = fitted.forecast(steps=n_eval_points)
        predicted_eval_spread = np.asarray(forecast_values, dtype=float)
    except Exception:
        return None

    if predicted_eval_spread.shape[0] != n_eval_points:
        return None

    rows: list[dict[str, float | str | pd.Timestamp]] = []
    max_origin_idx = n_eval_points - horizon - 1
    for i in range(0, max_origin_idx + 1):
        vol_at_origin = float(eval_vol[i])
        if not np.isfinite(vol_at_origin) or vol_at_origin <= 0.0:
            continue

        current_spread = float(eval_spread[i])
        current_beta = betas[i]

        predicted_future_spread = float(predicted_eval_spread[i + horizon])
        predicted_change = predicted_future_spread - current_spread
        predicted_z = predicted_change / vol_at_origin

        actual_future_spread = p1[i + horizon] - (current_beta * p2[i + horizon])
        actual_change = actual_future_spread - current_spread

        rows.append(
            {
                "forecast_origin_date": pd.to_datetime(eval_dates.iloc[i]),
                "target_date": pd.to_datetime(eval_dates.iloc[i + horizon]),
                "current_spread": current_spread,
                "actual_value": actual_future_spread,
                "predicted_value": predicted_future_spread,
                "actual_change": actual_change,
                "predicted_change": predicted_change,
                "rolling_vol_20d_at_origin": vol_at_origin,
                "predicted_z": predicted_z,
            }
        )

    if not rows:
        return None

    forecast_df = pd.DataFrame(rows).sort_values("forecast_origin_date").reset_index(drop=True)
    metrics_eval = _evaluate_fit_once_forecasts(forecast_df)
    metrics: dict[str, float | int | str] = {
        "pair": pair,
        "window_label": window_label,
        "eval_split": "val",
        "spread_col": spread_col,
        "p": int(p),
        "q": int(q),
        "horizon": int(horizon),
        "mse": metrics_eval["mse"],
        "mae": metrics_eval["mae"],
        "n_train": int(n_train),
        "n_eval_points": int(n_eval_points),
        "n_forecast_origins": int(len(forecast_df)),
        "mse_level": metrics_eval["mse_level"],
        "mae_level": metrics_eval["mae_level"],
    }
    return forecast_df, metrics


def run_tuning_for_window(
    window_dir: Path,
    spread_col: str,
    p_values: list[int],
    q_values: list[int],
    min_train_points: int,
    min_eval_points: int,
    horizon: int,
    target_pair: str | None = None,
) -> pd.DataFrame:
    """Run validation-only ARMA tuning for one window; returns successful rows."""

    splits = load_window_split_datasets(window_dir)
    window_label = splits.window_label

    if splits.val is None:
        warnings.warn(
            f"[{window_label}] No validation split found. Skipping this window.",
            stacklevel=2,
        )
        return pd.DataFrame()

    train_df = splits.train
    val_df = splits.val
    if spread_col not in train_df.columns or spread_col not in val_df.columns:
        raise ValueError(
            f"Spread column '{spread_col}' must exist in both train and val for window '{window_label}'."
        )
    _ = resolve_volatility_column(pd.concat([train_df, val_df], ignore_index=True), spread_col)

    pairs = _select_target_pairs(train_df, val_df, pair=target_pair)
    if not pairs:
        warnings.warn(f"[{window_label}] No eligible shared pairs for tuning.", stacklevel=2)
        return pd.DataFrame()

    print(f"\nWindow: {window_label}")
    print(f"  Candidate pairs: {len(pairs)}")
    print(f"  Grid size: {len(p_values)} x {len(q_values)}")

    all_rows: list[dict] = []

    for idx, pair in enumerate(pairs, start=1):
        print(f"  [{idx}/{len(pairs)}] Pair: {pair}")
        for p in p_values:
            for q in q_values:
                try:
                    result = fit_once_validate_pair(
                        train_df=train_df,
                        val_df=val_df,
                        pair=pair,
                        spread_col=spread_col,
                        p=int(p),
                        q=int(q),
                        horizon=horizon,
                        min_train_points=min_train_points,
                        min_eval_points=min_eval_points,
                        window_label=window_label,
                    )
                except Exception:
                    continue

                if result is None:
                    continue

                _, metrics = result
                all_rows.append(
                    {
                        "pair": metrics["pair"],
                        "window_label": metrics["window_label"],
                        "eval_split": "val",
                        "spread_col": metrics["spread_col"],
                        "p": int(metrics["p"]),
                        "q": int(metrics["q"]),
                        "horizon": int(metrics["horizon"]),
                        "mse": float(metrics["mse"]),
                        "mae": float(metrics["mae"]),
                        "n_train": int(metrics["n_train"]),
                        "n_eval_points": int(metrics["n_eval_points"]),
                        "n_forecast_origins": int(metrics["n_forecast_origins"]),
                        "mse_level": float(metrics["mse_level"]),
                        "mae_level": float(metrics["mae_level"]),
                    }
                )

    return pd.DataFrame(all_rows)


def aggregate_global_ranking(validation_results: pd.DataFrame) -> pd.DataFrame:
    """Aggregate validation rows by spread_col/p/q and apply global ranking order."""

    if validation_results.empty:
        return pd.DataFrame(
            columns=[
                "spread_col",
                "p",
                "q",
                "mean_val_mse",
                "mean_val_mae",
                "n_successful_runs",
                "n_windows",
                "n_pairs",
                "horizon",
            ]
        )

    ranking = (
        validation_results.groupby(["spread_col", "p", "q"], as_index=False)
        .agg(
            mean_val_mse=("mse", "mean"),
            mean_val_mae=("mae", "mean"),
            n_successful_runs=("pair", "size"),
            n_windows=("window_label", "nunique"),
            n_pairs=("pair", "nunique"),
            horizon=("horizon", "first"),
        )
        .copy()
    )

    ranking = ranking.sort_values(
        ["spread_col", "mean_val_mse", "mean_val_mae", "n_successful_runs", "p", "q"],
        ascending=[True, True, True, False, True, True],
    ).reset_index(drop=True)
    return ranking


def select_global_params(global_ranking: pd.DataFrame) -> pd.DataFrame:
    """Select one global (p, q) row per spread_col from ranked candidates."""

    if global_ranking.empty:
        return pd.DataFrame(
            columns=[
                "spread_col",
                "selected_p",
                "selected_q",
                "mean_val_mse",
                "mean_val_mae",
                "n_successful_runs",
                "n_windows",
                "n_pairs",
                "horizon",
                "selection_metric",
            ]
        )

    top_rows = global_ranking.groupby("spread_col", as_index=False).head(1).copy()
    top_rows = top_rows.rename(columns={"p": "selected_p", "q": "selected_q"})
    top_rows["selection_metric"] = "mse_change"

    return top_rows[
        [
            "spread_col",
            "selected_p",
            "selected_q",
            "mean_val_mse",
            "mean_val_mae",
            "n_successful_runs",
            "n_windows",
            "n_pairs",
            "horizon",
            "selection_metric",
        ]
    ].reset_index(drop=True)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Global ARMA tuning using validation MSE/MAE on 10-day spread change."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "pair_datasets"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "arma_tuning_outputs"),
    )
    parser.add_argument("--window", type=str, default=None, help="Optional single window label.")
    parser.add_argument("--pair", type=str, default=None, help="Optional single pair label.")
    parser.add_argument("--spread_col", type=str, default="spread_ols")
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=11)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--p_values", type=str, default=None, help="Comma-separated list, e.g. 1,2,3")
    parser.add_argument("--q_values", type=str, default=None, help="Comma-separated list, e.g. 1,2,3")
    args = parser.parse_args()

    p_values = _parse_int_list(args.p_values, P_VALUES)
    q_values = _parse_int_list(args.q_values, Q_VALUES)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / spread_variant_tag(args.spread_col)
    output_root.mkdir(parents=True, exist_ok=True)

    window_dirs = iter_window_dirs(input_root)
    if args.window is not None:
        window_dirs = [d for d in window_dirs if d.name == args.window]
        if not window_dirs:
            raise ValueError(f"Window '{args.window}' not found under {input_root}")

    all_validation_frames: list[pd.DataFrame] = []

    for window_dir in window_dirs:
        try:
            val_df = run_tuning_for_window(
                window_dir=window_dir,
                spread_col=args.spread_col,
                p_values=p_values,
                q_values=q_values,
                min_train_points=args.min_train_points,
                min_eval_points=args.min_eval_points,
                horizon=args.horizon,
                target_pair=args.pair,
            )
        except Exception as exc:
            warnings.warn(f"[{window_dir.name}] Tuning failed: {exc}", stacklevel=2)
            continue

        if not val_df.empty:
            all_validation_frames.append(val_df)

    all_validation_results = (
        pd.concat(all_validation_frames, ignore_index=True)
        if all_validation_frames
        else pd.DataFrame()
    )

    global_ranking = aggregate_global_ranking(all_validation_results)
    selected_global_params = select_global_params(global_ranking)

    all_validation_results.to_csv(output_root / "all_validation_results.csv", index=False)
    global_ranking.to_csv(output_root / "global_param_ranking.csv", index=False)
    selected_global_params.to_csv(output_root / "selected_global_params.csv", index=False)

    print("\n=== ARMA Global Tuning Complete ===")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Spread column: {args.spread_col}")
    print(f"Horizon: {args.horizon}")
    print(f"Grid size: {len(p_values)} x {len(q_values)}")
    print(f"Validation rows: {len(all_validation_results)}")
    print(f"Global candidates: {len(global_ranking)}")
    print(f"Selected rows: {len(selected_global_params)}")
    print(f"Saved: {output_root / 'all_validation_results.csv'}")
    print(f"Saved: {output_root / 'global_param_ranking.csv'}")
    print(f"Saved: {output_root / 'selected_global_params.csv'}")


if __name__ == "__main__":
    main()
