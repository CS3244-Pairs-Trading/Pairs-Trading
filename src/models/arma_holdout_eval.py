from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd
import numpy as np

from src.config import DEFAULT_CONFIG
from src.models.arma import (
    load_window_split_datasets,
    resolve_volatility_column,
    run_arma_for_pair,
    save_pair_outputs,
    spread_variant_tag,
)

REQUIRED_SELECTED_COLUMNS = {
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
}


def load_selected_global_params(
    tuning_root: Path,
    spread_col: str,
    horizon: int,
) -> pd.Series:
    """Load and validate selected_global_params.csv for one spread type."""

    path = tuning_root / "selected_global_params.csv"
    if not path.exists():
        raise FileNotFoundError(f"Selected global params file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_SELECTED_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"selected_global_params.csv missing required columns: {sorted(missing)}"
        )

    df = df[df["spread_col"].astype(str) == spread_col].copy()
    if df.empty:
        raise ValueError(
            f"No selected global parameter row found for spread_col='{spread_col}' in {path}"
        )

    row = df.iloc[0]
    selected_horizon = int(row["horizon"])
    if selected_horizon != int(horizon):
        raise ValueError(
            "Horizon mismatch between holdout run and selected global params: "
            f"requested={horizon}, selected={selected_horizon}."
        )

    return row


def evaluate_holdout_with_global_params(
    selected_row: pd.Series,
    input_root: Path,
    holdout_window: str,
    spread_col: str,
    min_train_points: int,
    min_eval_points: int,
    output_root: Path,
    save_forecasts: bool,
    horizon: int,
) -> pd.DataFrame:
    """Evaluate holdout test for all eligible pairs using one global (p, q)."""

    window_dir = input_root / holdout_window
    if not window_dir.exists():
        raise FileNotFoundError(f"Holdout window directory not found: {window_dir}")

    splits = load_window_split_datasets(window_dir)
    if splits.test is None:
        raise FileNotFoundError(
            f"Holdout test split not found for window '{holdout_window}'. "
            f"Expected file: {window_dir / 'test_pair_dataset.csv'}"
        )

    train_df = splits.train
    test_df = splits.test
    if spread_col not in train_df.columns or spread_col not in test_df.columns:
        raise ValueError(
            f"Spread column '{spread_col}' must exist in both holdout train and test for window '{holdout_window}'."
        )
    _ = resolve_volatility_column(pd.concat([train_df, test_df], ignore_index=True), spread_col)

    train_pairs = set(train_df["pair"].dropna().astype(str).unique())
    test_pairs = set(test_df["pair"].dropna().astype(str).unique())
    target_pairs = sorted(train_pairs.intersection(test_pairs))

    selected_p = int(selected_row["selected_p"])
    selected_q = int(selected_row["selected_q"])

    result_rows: list[dict] = []
    skipped = 0

    for pair in target_pairs:
        try:
            run = run_arma_for_pair(
                train_df=train_df,
                eval_df=test_df,
                pair=pair,
                spread_col=spread_col,
                p=selected_p,
                q=selected_q,
                min_train_points=min_train_points,
                min_eval_points=min_eval_points,
                eval_split="test",
                window_label=holdout_window,
                horizon=horizon,
            )
        except Exception:
            run = None

        if run is None:
            skipped += 1
            continue

        forecast_df, metrics, model_summary = run

        result_rows.append(
            {
                "pair": pair,
                "window_label": holdout_window,
                "eval_split": "test",
                "spread_col": spread_col,
                "selected_p": selected_p,
                "selected_q": selected_q,
                "horizon": int(horizon),
                "mse": float(metrics["mse"]),
                "rmse": float(np.sqrt(metrics["mse"])),
                "mae": float(metrics["mae"]),
                "n_train": int(metrics["n_train"]),
                "n_eval_points": int(metrics["n_eval_points"]),
                "n_forecast_origins": int(metrics["n_forecast_origins"]),
            }
        )

        if save_forecasts:
            save_pair_outputs(
                output_root=output_root,
                window_label=holdout_window,
                pair=pair,
                eval_split="test",
                forecast_df=forecast_df,
                metrics=result_rows[-1],
                fitted_model_summary=model_summary,
            )

    if skipped > 0:
        warnings.warn(f"Skipped {skipped} pairs during holdout evaluation.", stacklevel=2)

    return pd.DataFrame(result_rows)


def build_holdout_summary(
    holdout_results: pd.DataFrame,
    spread_col: str,
    holdout_window: str,
    selected_p: int,
    selected_q: int,
    horizon: int,
) -> pd.DataFrame:
    """Build one-row holdout summary."""

    if holdout_results.empty:
        mean_mse = float("nan")
        mean_mae = float("nan")
        n_pairs = 0
    else:
        mean_mse = float(holdout_results["mse"].mean())
        mean_rmse = float(holdout_results["rmse"].mean())
        mean_mae = float(holdout_results["mae"].mean())
        n_pairs = int(len(holdout_results))

    return pd.DataFrame(
        [
            {
                "spread_col": spread_col,
                "holdout_window": holdout_window,
                "selected_p": int(selected_p),
                "selected_q": int(selected_q),
                "horizon": int(horizon),
                "mean_mse": mean_mse,
                "mean_rmse": mean_rmse,
                "mean_mae": mean_mae,
                "n_pairs_evaluated": n_pairs,
            }
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Evaluate ARMA on holdout test split using globally selected parameters."
    )
    parser.add_argument(
        "--tuning_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "arma_tuning_outputs"),
    )
    parser.add_argument(
        "--input_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "pair_datasets"),
    )
    parser.add_argument(
        "--output_root",
        type=str,
        default=str(DEFAULT_CONFIG.processed_dir / "arma_holdout_outputs"),
    )
    parser.add_argument(
        "--holdout_window",
        type=str,
        default=DEFAULT_CONFIG.holdout_split.label,
    )
    parser.add_argument("--spread_col", type=str, default="spread_ols")
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=11)
    parser.add_argument("--save_forecasts", action="store_true")
    parser.add_argument("--horizon", type=int, default=10)
    args = parser.parse_args()

    input_root = Path(args.input_root)
    tuning_root = Path(args.tuning_root) / spread_variant_tag(args.spread_col)
    output_root = Path(args.output_root) / spread_variant_tag(args.spread_col)
    output_root.mkdir(parents=True, exist_ok=True)

    selected_row = load_selected_global_params(
        tuning_root=tuning_root,
        spread_col=args.spread_col,
        horizon=args.horizon,
    )

    holdout_results = evaluate_holdout_with_global_params(
        selected_row=selected_row,
        input_root=input_root,
        holdout_window=args.holdout_window,
        spread_col=args.spread_col,
        min_train_points=args.min_train_points,
        min_eval_points=args.min_eval_points,
        output_root=output_root,
        save_forecasts=args.save_forecasts,
        horizon=args.horizon,
    )

    selected_used_df = pd.DataFrame([selected_row.to_dict()])
    holdout_summary = build_holdout_summary(
        holdout_results=holdout_results,
        spread_col=args.spread_col,
        holdout_window=args.holdout_window,
        selected_p=int(selected_row["selected_p"]),
        selected_q=int(selected_row["selected_q"]),
        horizon=args.horizon,
    )

    holdout_results.to_csv(output_root / "final_holdout_results.csv", index=False)
    holdout_summary.to_csv(output_root / "holdout_summary.csv", index=False)
    selected_used_df.to_csv(output_root / "selected_global_params_used.csv", index=False)

    print("\n=== ARMA Holdout Evaluation Complete ===")
    print(f"Tuning root: {tuning_root}")
    print(f"Pair dataset root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Holdout window: {args.holdout_window}")
    print(f"Spread column: {args.spread_col}")
    print(f"Selected global p,q: ({int(selected_row['selected_p'])}, {int(selected_row['selected_q'])})")
    print(f"Pairs evaluated: {len(holdout_results)}")
    print(f"Saved: {output_root / 'final_holdout_results.csv'}")
    print(f"Saved: {output_root / 'holdout_summary.csv'}")
    print(f"Saved: {output_root / 'selected_global_params_used.csv'}")


if __name__ == "__main__":
    main()
