from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from src.config import DEFAULT_CONFIG

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class WindowSplitDatasets:
    """Container for one window's available split datasets."""

    window_label: str
    train: pd.DataFrame
    val: pd.DataFrame | None
    test: pd.DataFrame | None


def iter_window_dirs(root: Path) -> list[Path]:
    """Return sorted window directories under the pair dataset root."""

    if not root.exists():
        raise FileNotFoundError(f"Input root does not exist: {root}")
    if not root.is_dir():
        raise ValueError(f"Input root is not a directory: {root}")
    return sorted([p for p in root.iterdir() if p.is_dir()], key=lambda p: p.name)


def load_pair_dataset(file_path: str | Path, date_col: str = "Date") -> pd.DataFrame:
    """Load one pair dataset CSV and validate required columns."""

    path = Path(file_path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset file not found: {path}")

    df = pd.read_csv(path)
    required = {date_col, "pair"}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(
            f"Dataset '{path}' missing required columns: {', '.join(sorted(missing))}"
        )

    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df = df.dropna(subset=[date_col]).copy()
    return df.sort_values(date_col).reset_index(drop=True)


def load_window_split_datasets(window_dir: Path) -> WindowSplitDatasets:
    """
    Load train/val/test datasets for one window folder.

    Expected files:
    - required: train_pair_dataset.csv
    - optional: val_pair_dataset.csv
    - optional: test_pair_dataset.csv
    """

    train_path = window_dir / "train_pair_dataset.csv"
    val_path = window_dir / "val_pair_dataset.csv"
    test_path = window_dir / "test_pair_dataset.csv"

    if not train_path.exists():
        raise FileNotFoundError(f"Missing train split for '{window_dir.name}': {train_path}")

    train_df = load_pair_dataset(train_path)
    val_df = load_pair_dataset(val_path) if val_path.exists() else None
    test_df = load_pair_dataset(test_path) if test_path.exists() else None

    return WindowSplitDatasets(
        window_label=window_dir.name,
        train=train_df,
        val=val_df,
        test=test_df,
    )


def spread_variant_tag(spread_col: str) -> str:
    """Map spread column to output variant folder."""

    if spread_col == "spread_ols":
        return "arma_ols"
    if spread_col == "spread_kalman":
        return "arma_kalman"
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", spread_col.strip())
    safe = safe.strip("._") or "spread"
    return f"arma_{safe}"


def resolve_volatility_column(df: pd.DataFrame, spread_col: str) -> str:
    """
    Resolve the volatility column used to derive predicted z-scores.

    For Kalman spreads, prefer a Kalman-specific 20d volatility column when it
    exists. If none is available, fall back to the generic rolling_vol_20d so
    ARMA can still run and tune on spread_kalman. The volatility column is only
    used to derive predicted_z; it is not used in the ARMA fit itself.
    """

    generic_vol = "rolling_vol_20d"

    if spread_col == "spread_ols":
        if generic_vol not in df.columns:
            raise ValueError(
                "spread_ols requires volatility column 'rolling_vol_20d', "
                f"but it was not found. Available columns: {list(df.columns)[:30]}"
            )
        return generic_vol

    if spread_col == "spread_kalman":
        preferred = "rolling_vol_20d_kalman"
        if preferred in df.columns:
            return preferred

        kalman_vol_candidates = sorted(
            {
                c
                for c in df.columns
                if "rolling_vol" in c.lower() and "kalman" in c.lower()
            }
        )
        if len(kalman_vol_candidates) == 1:
            return kalman_vol_candidates[0]
        if len(kalman_vol_candidates) > 1:
            raise ValueError(
                "Multiple Kalman volatility columns found. "
                f"Candidates: {kalman_vol_candidates}. Please keep one canonical column."
            )

        if generic_vol in df.columns:
            warnings.warn(
                "No Kalman-specific volatility column found for spread_kalman; "
                "falling back to generic 'rolling_vol_20d' to derive predicted_z.",
                stacklevel=2,
            )
            return generic_vol

        vol_like_cols = [c for c in df.columns if "vol" in c.lower()]
        raise ValueError(
            "spread_kalman requires a volatility column for predicted_z. Looked for "
            "'rolling_vol_20d_kalman', any column containing both 'rolling_vol' and "
            f"'kalman', and finally fallback 'rolling_vol_20d'. None found. Vol-like columns: {vol_like_cols}"
        )

    raise ValueError(
        f"Unsupported spread_col '{spread_col}'. Expected 'spread_ols' or 'spread_kalman'."
    )


def extract_pair_spread(df: pd.DataFrame, pair: str, spread_col: str) -> pd.Series:
    """Extract one pair's spread as Date-indexed series. Empty series if no rows."""

    if spread_col not in df.columns:
        raise ValueError(
            f"Spread column '{spread_col}' not found. "
            f"Available columns (first 25): {list(df.columns)[:25]}"
        )

    out = df.loc[df["pair"].astype(str) == str(pair), ["Date", spread_col]].copy()
    out = out.dropna(subset=[spread_col]).sort_values("Date")
    if out.empty:
        return pd.Series(dtype=float, name="spread")

    series = out.set_index("Date")[spread_col].astype(float)
    series.name = "spread"
    return series


def extract_pair_frame(df: pd.DataFrame, pair: str, spread_col: str, vol_col: str) -> pd.DataFrame:
    """Extract Date/spread/volatility rows for one pair, sorted by Date."""

    needed = {"Date", "pair", spread_col, vol_col}
    missing = needed - set(df.columns)
    if missing:
        raise ValueError(f"Missing required columns for pair extraction: {sorted(missing)}")

    out = df.loc[df["pair"].astype(str) == str(pair), ["Date", spread_col, vol_col]].copy()
    if out.empty:
        return out

    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    out[spread_col] = pd.to_numeric(out[spread_col], errors="coerce")
    out[vol_col] = pd.to_numeric(out[vol_col], errors="coerce")

    out = out.dropna(subset=["Date", spread_col]).sort_values("Date").reset_index(drop=True)
    return out


def compute_mse(actual: pd.Series, predicted: pd.Series) -> float:
    diff = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.mean(diff**2))


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    diff = np.abs(actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float))
    return float(np.mean(diff))


def forecast_horizon_walk_forward(
    train_pair_df: pd.DataFrame,
    eval_pair_df: pd.DataFrame,
    spread_col: str,
    vol_col: str,
    p: int,
    q: int,
    horizon: int = 10,
) -> pd.DataFrame:
    """
    Forecast one pair with no leakage using expanding history up to each origin.

    At each origin t in eval data, fit ARMA on train + eval[:t] and forecast horizon steps,
    then keep only the horizon-th value aligned to eval[t + horizon].
    """

    if horizon <= 0:
        raise ValueError("horizon must be > 0")

    train_work = train_pair_df[["Date", spread_col]].copy()
    eval_work = eval_pair_df[["Date", spread_col, vol_col]].copy()

    if train_work.empty or eval_work.empty:
        return pd.DataFrame()

    n_eval = len(eval_work)
    max_origin_idx = n_eval - horizon - 1
    if max_origin_idx < 0:
        return pd.DataFrame()

    rows: list[dict[str, float | str | pd.Timestamp]] = []

    for origin_idx in range(0, max_origin_idx + 1):
        origin_row = eval_work.iloc[origin_idx]
        target_row = eval_work.iloc[origin_idx + horizon]

        history_series = pd.concat(
            [train_work[spread_col], eval_work.iloc[: origin_idx + 1][spread_col]],
            ignore_index=True,
        ).astype(float)

        if history_series.empty or history_series.isna().any():
            continue

        rolling_vol_at_origin = float(origin_row[vol_col])
        if not np.isfinite(rolling_vol_at_origin) or rolling_vol_at_origin <= 0.0:
            continue

        try:
            fitted = ARIMA(history_series, order=(p, 0, q)).fit()
            forecast_values = fitted.forecast(steps=horizon)
            predicted_future_spread = float(np.asarray(forecast_values, dtype=float)[-1])
        except Exception:
            continue

        current_spread = float(origin_row[spread_col])
        actual_future_spread = float(target_row[spread_col])
        actual_change = actual_future_spread - current_spread
        predicted_change = predicted_future_spread - current_spread
        predicted_z = predicted_change / rolling_vol_at_origin

        rows.append(
            {
                "forecast_origin_date": pd.to_datetime(origin_row["Date"]),
                "target_date": pd.to_datetime(target_row["Date"]),
                "current_spread": current_spread,
                "actual_future_spread": actual_future_spread,
                "predicted_future_spread": predicted_future_spread,
                "actual_change_10d": actual_change,
                "predicted_change_10d": predicted_change,
                "rolling_vol_20d_at_origin": rolling_vol_at_origin,
                "predicted_z_10d": predicted_z,
            }
        )

    if not rows:
        return pd.DataFrame()

    return pd.DataFrame(rows).sort_values("forecast_origin_date").reset_index(drop=True)


def evaluate_forecasts(forecast_df: pd.DataFrame) -> dict[str, float]:
    """Official layer-1 metrics are MSE/MAE on change forecasts."""

    actual_change = forecast_df["actual_change"].astype(float)
    predicted_change = forecast_df["predicted_change"].astype(float)
    actual_value = forecast_df["actual_value"].astype(float)
    predicted_value = forecast_df["predicted_value"].astype(float)

    return {
        "mse": compute_mse(actual_change, predicted_change),
        "mae": compute_mae(actual_change, predicted_change),
        "mse_level": compute_mse(actual_value, predicted_value),
        "mae_level": compute_mae(actual_value, predicted_value),
    }


def _safe_pair_name(pair: str) -> str:
    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.strip())
    return safe.strip("._") or "pair"


def run_arma_for_pair(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pair: str,
    spread_col: str,
    p: int,
    q: int,
    min_train_points: int,
    min_eval_points: int,
    eval_split: str,
    window_label: str,
    horizon: int = 10,
) -> tuple[pd.DataFrame, dict[str, float | int | str], str] | None:
    """Run ARMA walk-forward forecast for one pair and split. Returns None when skipped."""

    if spread_col not in train_df.columns or spread_col not in eval_df.columns:
        raise ValueError(
            f"Spread column '{spread_col}' must exist in both train and eval splits."
        )

    vol_col = resolve_volatility_column(pd.concat([train_df, eval_df], ignore_index=True), spread_col)
    train_pair_df = extract_pair_frame(train_df, pair=pair, spread_col=spread_col, vol_col=vol_col)
    eval_pair_df = extract_pair_frame(eval_df, pair=pair, spread_col=spread_col, vol_col=vol_col)

    n_train = len(train_pair_df)
    n_eval_points = len(eval_pair_df)

    if n_train < min_train_points:
        return None
    if n_eval_points < max(min_eval_points, horizon + 1):
        return None

    walk_df = forecast_horizon_walk_forward(
        train_pair_df=train_pair_df,
        eval_pair_df=eval_pair_df,
        spread_col=spread_col,
        vol_col=vol_col,
        p=p,
        q=q,
        horizon=horizon,
    )
    if walk_df.empty:
        return None

    forecast_df = walk_df.rename(
        columns={
            "actual_future_spread": "actual_value",
            "predicted_future_spread": "predicted_value",
            "actual_change_10d": "actual_change",
            "predicted_change_10d": "predicted_change",
            "predicted_z_10d": "predicted_z",
        }
    )
    forecast_df["pair"] = pair
    forecast_df["spread_col"] = spread_col
    forecast_df["p"] = int(p)
    forecast_df["q"] = int(q)
    forecast_df["horizon"] = int(horizon)
    forecast_df["eval_split"] = eval_split
    forecast_df["window_label"] = window_label

    final_cols = [
        "forecast_origin_date",
        "target_date",
        "current_spread",
        "actual_value",
        "predicted_value",
        "actual_change",
        "predicted_change",
        "rolling_vol_20d_at_origin",
        "predicted_z",
        "pair",
        "spread_col",
        "p",
        "q",
        "horizon",
        "eval_split",
        "window_label",
    ]
    forecast_df = forecast_df[final_cols].copy()

    eval_metrics = evaluate_forecasts(forecast_df)
    metrics: dict[str, float | int | str] = {
        "pair": pair,
        "spread_col": spread_col,
        "p": int(p),
        "q": int(q),
        "horizon": int(horizon),
        "n_train": int(n_train),
        "n_eval_points": int(n_eval_points),
        "n_forecast_origins": int(len(forecast_df)),
        "eval_split": eval_split,
        "window_label": window_label,
        "mse": eval_metrics["mse"],
        "mae": eval_metrics["mae"],
        "mse_level": eval_metrics["mse_level"],
        "mae_level": eval_metrics["mae_level"],
    }

    model_summary = (
        "Walk-forward ARMA via ARIMA(order=(p,0,q)); model refit per forecast origin. "
        f"pair={pair}, spread_col={spread_col}, vol_col={vol_col}, p={p}, q={q}, horizon={horizon}, "
        f"origins={len(forecast_df)}"
    )
    return forecast_df, metrics, model_summary


def save_pair_outputs(
    output_root: Path,
    window_label: str,
    pair: str,
    eval_split: str,
    forecast_df: pd.DataFrame,
    metrics: dict[str, float | int | str],
    fitted_model_summary: str,
) -> None:
    """Save one pair's outputs with eval-split-specific filenames."""

    pair_dir = output_root / window_label / "pairs" / _safe_pair_name(pair)
    pair_dir.mkdir(parents=True, exist_ok=True)

    forecast_df.to_csv(pair_dir / f"arma_forecasts_{eval_split}.csv", index=False)
    pd.DataFrame([metrics]).to_csv(pair_dir / f"arma_metrics_{eval_split}.csv", index=False)
    (pair_dir / f"arma_model_summary_{eval_split}.txt").write_text(fitted_model_summary)


def build_shared_predictions_export(forecast_df: pd.DataFrame) -> pd.DataFrame:
    """
    Build the normalized cross-model predictions export from ARMA detailed forecasts.

    ARMA keeps richer detailed outputs; this export is the shared backtest interface.
    """

    required = {
        "forecast_origin_date",
        "pair",
        "predicted_change",
        "predicted_value",
        "predicted_z",
    }
    missing = required - set(forecast_df.columns)
    if missing:
        raise ValueError(f"Cannot build shared predictions export; missing columns: {sorted(missing)}")

    shared = forecast_df.rename(columns={"forecast_origin_date": "Date"})[
        ["Date", "pair", "predicted_change", "predicted_value", "predicted_z"]
    ].copy()
    return shared


def save_shared_predictions_export(
    forecast_df: pd.DataFrame,
    output_path: Path,
) -> None:
    """Save normalized predictions CSV sorted by pair/date with (Date, pair) dedup only."""

    shared = build_shared_predictions_export(forecast_df)
    shared = shared.sort_values(["pair", "Date"]).drop_duplicates(
        subset=["Date", "pair"], keep="first"
    )
    shared.to_csv(output_path, index=False)


def save_window_outputs(
    output_root: Path,
    window_label: str,
    all_forecasts: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """Save aggregate outputs for one window."""

    window_dir = output_root / window_label
    window_dir.mkdir(parents=True, exist_ok=True)

    all_forecasts = all_forecasts.sort_values(
        ["eval_split", "pair", "forecast_origin_date"]
    ).reset_index(drop=True)
    metrics_df = metrics_df.sort_values(["eval_split", "mse", "mae", "pair"]).reset_index(drop=True)

    all_forecasts.to_csv(window_dir / "all_forecasts.csv", index=False)
    metrics_df.to_csv(window_dir / "summary_metrics.csv", index=False)
    save_shared_predictions_export(
        forecast_df=all_forecasts,
        output_path=window_dir / "predictions.csv",
    )

    split_labels = sorted(metrics_df["eval_split"].astype(str).unique())
    for split in split_labels:
        split_forecasts = all_forecasts[all_forecasts["eval_split"] == split]
        split_metrics = metrics_df[metrics_df["eval_split"] == split]
        split_forecasts.to_csv(window_dir / f"all_forecasts_{split}.csv", index=False)
        split_metrics.to_csv(window_dir / f"summary_metrics_{split}.csv", index=False)

        if len(split_labels) > 1:
            save_shared_predictions_export(
                forecast_df=split_forecasts,
                output_path=window_dir / f"predictions_{split}.csv",
            )


def _resolve_eval_datasets(
    splits: WindowSplitDatasets,
    eval_split_arg: str,
) -> list[tuple[str, pd.DataFrame]]:
    """Resolve which eval datasets to use based on CLI option and availability."""

    available: list[tuple[str, pd.DataFrame]] = []
    if splits.val is not None:
        available.append(("val", splits.val))
    if splits.test is not None:
        available.append(("test", splits.test))
    if not available:
        return []

    if eval_split_arg == "val":
        return [("val", splits.val)] if splits.val is not None else []
    if eval_split_arg == "test":
        return [("test", splits.test)] if splits.test is not None else []
    if eval_split_arg == "both":
        return available

    if splits.val is not None:
        return [("val", splits.val)]
    return [("test", splits.test)] if splits.test is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-parameter ARMA across pair-dataset windows (layer-1 only)."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "pair_datasets"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "predictions"),
    )
    parser.add_argument("--window", type=str, default=None, help="Optional single window label.")
    parser.add_argument("--pair", type=str, default=None, help="Optional single pair label.")
    parser.add_argument("--spread_col", type=str, default="spread_ols")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--horizon", type=int, default=10)
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=11)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="auto",
        choices=["auto", "val", "test", "both"],
        help="Which evaluation split to run. 'auto' prefers val, otherwise test.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root) / spread_variant_tag(args.spread_col)
    output_root.mkdir(parents=True, exist_ok=True)

    window_dirs = iter_window_dirs(input_root)
    if args.window is not None:
        window_dirs = [p for p in window_dirs if p.name == args.window]
        if not window_dirs:
            raise ValueError(f"Window '{args.window}' not found under {input_root}")

    total_modeled = 0
    total_skipped = 0
    processed_windows = 0

    for window_dir in window_dirs:
        window_label = window_dir.name
        try:
            splits = load_window_split_datasets(window_dir)
        except Exception as exc:
            warnings.warn(f"[{window_label}] Failed to load splits: {exc}", stacklevel=2)
            continue

        eval_datasets = _resolve_eval_datasets(splits, args.eval_split)
        if not eval_datasets:
            warnings.warn(
                f"[{window_label}] No eval split available for option '{args.eval_split}'.",
                stacklevel=2,
            )
            continue

        window_forecasts: list[pd.DataFrame] = []
        window_metrics: list[dict[str, float | int | str]] = []
        modeled = 0
        skipped = 0

        for eval_name, eval_df in eval_datasets:
            _ = resolve_volatility_column(
                pd.concat([splits.train, eval_df], ignore_index=True),
                args.spread_col,
            )
            if args.spread_col not in splits.train.columns or args.spread_col not in eval_df.columns:
                raise ValueError(
                    f"Spread column '{args.spread_col}' must exist in both train and {eval_name}."
                )

            train_pairs = set(splits.train["pair"].dropna().astype(str).unique())
            eval_pairs = set(eval_df["pair"].dropna().astype(str).unique())
            shared_pairs = sorted(train_pairs.intersection(eval_pairs))

            if args.pair is not None:
                if args.pair not in shared_pairs:
                    warnings.warn(
                        f"[{window_label}:{eval_name}] Pair '{args.pair}' not in both train/eval.",
                        stacklevel=2,
                    )
                    continue
                target_pairs = [args.pair]
            else:
                target_pairs = shared_pairs

            if not target_pairs:
                warnings.warn(
                    f"[{window_label}:{eval_name}] No shared pairs between train and eval.",
                    stacklevel=2,
                )
                continue

            print(f"\nWindow: {window_label} | Split: {eval_name} | Candidate pairs: {len(target_pairs)}")

            for idx, pair in enumerate(target_pairs, start=1):
                print(f"  [{idx}/{len(target_pairs)}] {pair}")
                try:
                    result = run_arma_for_pair(
                        train_df=splits.train,
                        eval_df=eval_df,
                        pair=pair,
                        spread_col=args.spread_col,
                        p=args.p,
                        q=args.q,
                        min_train_points=args.min_train_points,
                        min_eval_points=args.min_eval_points,
                        eval_split=eval_name,
                        window_label=window_label,
                        horizon=args.horizon,
                    )
                except Exception as exc:
                    warnings.warn(
                        f"[{window_label}:{eval_name}:{pair}] Model failed: {exc}",
                        stacklevel=2,
                    )
                    skipped += 1
                    continue

                if result is None:
                    skipped += 1
                    continue

                forecast_df, metrics, summary = result
                save_pair_outputs(
                    output_root=output_root,
                    window_label=window_label,
                    pair=pair,
                    eval_split=eval_name,
                    forecast_df=forecast_df,
                    metrics=metrics,
                    fitted_model_summary=summary,
                )
                window_forecasts.append(forecast_df)
                window_metrics.append(metrics)
                modeled += 1

        if modeled == 0:
            warnings.warn(f"[{window_label}] No pairs successfully modeled.", stacklevel=2)
            continue

        forecasts_df = pd.concat(window_forecasts, ignore_index=True)
        metrics_df = pd.DataFrame(window_metrics)
        save_window_outputs(
            output_root=output_root,
            window_label=window_label,
            all_forecasts=forecasts_df,
            metrics_df=metrics_df,
        )

        processed_windows += 1
        total_modeled += modeled
        total_skipped += skipped
        print(f"  Modeled: {modeled} | Skipped: {skipped}")
        print(f"  Output: {output_root / window_label}")

    if processed_windows == 0:
        raise ValueError("No windows were processed successfully.")

    print("\n=== Fixed ARMA Run Complete ===")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Spread column: {args.spread_col}")
    print(f"Processed windows: {processed_windows}")
    print(f"Total modeled pairs: {total_modeled}")
    print(f"Total skipped pairs: {total_skipped}")


if __name__ == "__main__":
    main()
