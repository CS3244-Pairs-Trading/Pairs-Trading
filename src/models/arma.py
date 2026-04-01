from __future__ import annotations

import argparse
import re
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

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


def compute_rmse(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute root mean squared error."""

    diff = actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float)
    return float(np.sqrt(np.mean(diff**2)))


def compute_mae(actual: pd.Series, predicted: pd.Series) -> float:
    """Compute mean absolute error."""

    diff = np.abs(actual.to_numpy(dtype=float) - predicted.to_numpy(dtype=float))
    return float(np.mean(diff))


def fit_arma_model(train_series: pd.Series, p: int, q: int):
    """Fit ARMA(p, q) via ARIMA(p, 0, q)."""

    model = ARIMA(train_series, order=(p, 0, q))
    return model.fit()


def forecast_horizon_once(
    fitted_model,
    steps: int,
    forecast_index: pd.Index,
) -> pd.Series:
    """Forecast the full horizon in one call using a fitted ARMA model."""

    if steps <= 0:
        raise ValueError("steps must be > 0 for horizon forecasting.")
    preds = fitted_model.forecast(steps=steps)
    return pd.Series(np.asarray(preds, dtype=float), index=forecast_index, name="predicted")


def rolling_forecast(
    train_series: pd.Series,
    eval_series: pd.Series,
    p: int,
    q: int,
) -> pd.DataFrame:
    """
    Rolling one-step-ahead ARMA forecast over eval series.

    Re-fits each step and appends actual eval value to history.
    """

    history = list(train_series.values)
    predictions: list[float] = []

    for actual in eval_series.values:
        model = ARIMA(history, order=(p, 0, q))
        fitted = model.fit()
        pred = fitted.forecast(steps=1)[0]
        predictions.append(float(pred))
        history.append(float(actual))

    return pd.DataFrame(
        {
            "Date": pd.to_datetime(eval_series.index),
            "actual": eval_series.values,
            "predicted": predictions,
        }
    )


def evaluate_forecasts(results: pd.DataFrame) -> dict[str, float]:
    """Evaluate forecast dataframe using RMSE (primary) and MAE (secondary)."""

    rmse = compute_rmse(results["actual"], results["predicted"])
    mae = compute_mae(results["actual"], results["predicted"])
    return {"rmse": rmse, "mae": mae}


def evaluate_series_forecasts(actual: pd.Series, predicted: pd.Series) -> dict[str, float]:
    """Evaluate aligned series forecasts using RMSE (primary) and MAE (secondary)."""

    return {
        "rmse": compute_rmse(actual, predicted),
        "mae": compute_mae(actual, predicted),
    }


def generate_trading_signals(
    results: pd.DataFrame,
    entry_z: float = 2.0,
    exit_z: float = 0.5,
) -> pd.DataFrame:
    """Generate simple z-score-based long/short/flat position signals."""

    df = results.copy()
    df["forecast_error"] = df["actual"] - df["predicted"]

    error_mean = df["forecast_error"].mean()
    error_std = df["forecast_error"].std()
    if error_std == 0 or np.isnan(error_std):
        df["zscore"] = 0.0
    else:
        df["zscore"] = (df["forecast_error"] - error_mean) / error_std

    position = 0
    positions: list[int] = []
    for z in df["zscore"]:
        if position == 0:
            if z > entry_z:
                position = -1
            elif z < -entry_z:
                position = 1
        else:
            if abs(z) < exit_z:
                position = 0
        positions.append(position)

    df["position"] = positions
    return df


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


def _safe_pair_name(pair: str) -> str:
    """Sanitize pair names for filesystem paths."""

    safe = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.strip())
    return safe.strip("._") or "pair"


def run_arma_for_pair(
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    pair: str,
    spread_col: str,
    p: int,
    q: int,
    entry_z: float,
    exit_z: float,
    min_train_points: int,
    min_eval_points: int,
    eval_split: str,
    window_label: str,
    forecast_mode: Literal["rolling", "once"] = "rolling",
) -> tuple[pd.DataFrame, dict[str, float | int | str], str] | None:
    """Run ARMA for one pair and one eval split. Returns None when skipped."""

    train_series = extract_pair_spread(train_df, pair=pair, spread_col=spread_col)
    eval_series = extract_pair_spread(eval_df, pair=pair, spread_col=spread_col)

    if train_series.empty or eval_series.empty:
        return None
    if len(train_series) < min_train_points or len(eval_series) < min_eval_points:
        return None

    fitted_model = fit_arma_model(train_series=train_series, p=p, q=q)
    if forecast_mode == "rolling":
        forecast_df = rolling_forecast(train_series=train_series, eval_series=eval_series, p=p, q=q)
    elif forecast_mode == "once":
        predicted = forecast_horizon_once(
            fitted_model=fitted_model,
            steps=len(eval_series),
            forecast_index=eval_series.index,
        )
        forecast_df = pd.DataFrame(
            {
                "Date": pd.to_datetime(eval_series.index),
                "actual": eval_series.values,
                "predicted": predicted.values,
            }
        )
    else:
        raise ValueError(f"Unsupported forecast_mode '{forecast_mode}'.")

    forecast_df = generate_trading_signals(forecast_df, entry_z=entry_z, exit_z=exit_z)
    forecast_df["pair"] = pair
    forecast_df["spread_col"] = spread_col
    forecast_df["p"] = p
    forecast_df["q"] = q
    forecast_df["eval_split"] = eval_split
    forecast_df["window_label"] = window_label

    metrics = evaluate_forecasts(forecast_df)
    metrics.update(
        {
            "pair": pair,
            "spread_col": spread_col,
            "p": p,
            "q": q,
            "n_train": len(train_series),
            "n_eval": len(eval_series),
            "eval_split": eval_split,
            "window_label": window_label,
        }
    )
    return forecast_df, metrics, str(fitted_model.summary())


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


def save_window_outputs(
    output_root: Path,
    window_label: str,
    all_forecasts: pd.DataFrame,
    metrics_df: pd.DataFrame,
) -> None:
    """Save aggregate outputs for one window."""

    window_dir = output_root / window_label
    window_dir.mkdir(parents=True, exist_ok=True)
    all_forecasts.to_csv(window_dir / "all_forecasts.csv", index=False)
    metrics_df.to_csv(window_dir / "summary_metrics.csv", index=False)

    for split in sorted(metrics_df["eval_split"].astype(str).unique()):
        split_forecasts = all_forecasts[all_forecasts["eval_split"] == split]
        split_metrics = metrics_df[metrics_df["eval_split"] == split]
        split_forecasts.to_csv(window_dir / f"all_forecasts_{split}.csv", index=False)
        split_metrics.to_csv(window_dir / f"summary_metrics_{split}.csv", index=False)


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

    # auto
    if splits.val is not None:
        return [("val", splits.val)]
    return [("test", splits.test)] if splits.test is not None else []


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run fixed-parameter ARMA across pair-dataset windows."
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "pair_datasets"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "arma_outputs"),
    )
    parser.add_argument("--window", type=str, default=None, help="Optional single window label.")
    parser.add_argument("--pair", type=str, default=None, help="Optional single pair label.")
    parser.add_argument("--spread_col", type=str, default="spread_ols")
    parser.add_argument("--p", type=int, default=1)
    parser.add_argument("--q", type=int, default=1)
    parser.add_argument("--entry_z", type=float, default=2.0)
    parser.add_argument("--exit_z", type=float, default=0.5)
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=10)
    parser.add_argument(
        "--eval_split",
        type=str,
        default="auto",
        choices=["auto", "val", "test", "both"],
        help="Which evaluation split to run. 'auto' prefers val, otherwise test.",
    )
    args = parser.parse_args()

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)

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
            warnings.warn(f"[{window_label}] No eval split available for option '{args.eval_split}'.", stacklevel=2)
            continue

        window_forecasts: list[pd.DataFrame] = []
        window_metrics: list[dict[str, float | int | str]] = []
        modeled = 0
        skipped = 0

        for eval_name, eval_df in eval_datasets:
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
                        entry_z=args.entry_z,
                        exit_z=args.exit_z,
                        min_train_points=args.min_train_points,
                        min_eval_points=args.min_eval_points,
                        eval_split=eval_name,
                        window_label=window_label,
                    )
                except Exception as exc:
                    warnings.warn(f"[{window_label}:{eval_name}:{pair}] Model failed: {exc}", stacklevel=2)
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

        forecasts_df = pd.concat(window_forecasts, ignore_index=True).sort_values(
            ["eval_split", "pair", "Date"]
        ).reset_index(drop=True)
        metrics_df = pd.DataFrame(window_metrics).sort_values(
            ["eval_split", "rmse", "pair"]
        ).reset_index(drop=True)
        save_window_outputs(output_root=output_root, window_label=window_label, all_forecasts=forecasts_df, metrics_df=metrics_df)

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
