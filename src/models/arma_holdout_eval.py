from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_CONFIG
from src.models.arma import (
    load_window_split_datasets,
    run_arma_for_pair,
    save_pair_outputs,
)


REQUIRED_TUNING_COLUMNS = {"pair", "p", "q", "rmse", "mae", "window_label"}


def load_validation_tuning_results(
    tuning_root: Path,
    spread_col: str,
) -> pd.DataFrame:
    """Load and validate all_validation_results.csv used for parameter selection."""

    path = tuning_root / "all_validation_results.csv"
    if not path.exists():
        raise FileNotFoundError(f"Tuning results file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_TUNING_COLUMNS - set(df.columns)
    if missing:
        raise ValueError(
            f"all_validation_results.csv missing required columns: {sorted(missing)}"
        )

    if "eval_split" in df.columns:
        df = df[df["eval_split"].astype(str) == "val"].copy()
    if "spread_col" in df.columns:
        df = df[df["spread_col"].astype(str) == spread_col].copy()

    if df.empty:
        raise ValueError(
            "No validation tuning rows available after filtering. "
            "Check spread_col or tuning outputs."
        )

    return df


def select_final_params_per_pair(
    validation_df: pd.DataFrame,
    spread_col: str,
) -> pd.DataFrame:
    """
    Select one final (p, q) per pair from aggregated validation performance.

    Selection order (best first):
    1) lowest mean_val_rmse
    2) lowest mean_val_mae
    3) largest n_tuning_windows
    4) smaller p
    5) smaller q
    """

    agg = (
        validation_df.groupby(["pair", "p", "q"], as_index=False)
        .agg(
            mean_val_rmse=("rmse", "mean"),
            mean_val_mae=("mae", "mean"),
            n_tuning_windows=("window_label", "nunique"),
        )
        .copy()
    )

    ordered = agg.sort_values(
        ["pair", "mean_val_rmse", "mean_val_mae", "n_tuning_windows", "p", "q"],
        ascending=[True, True, True, False, True, True],
    )
    selected = ordered.groupby("pair", as_index=False).head(1).reset_index(drop=True)
    selected = selected.rename(columns={"p": "selected_p", "q": "selected_q"})
    selected["spread_col"] = spread_col
    return selected[
        [
            "pair",
            "selected_p",
            "selected_q",
            "mean_val_rmse",
            "mean_val_mae",
            "n_tuning_windows",
            "spread_col",
        ]
    ]


def evaluate_holdout_with_selected_params(
    selected_params_df: pd.DataFrame,
    input_root: Path,
    holdout_window: str,
    spread_col: str,
    entry_z: float,
    exit_z: float,
    min_train_points: int,
    min_eval_points: int,
    output_root: Path,
    save_forecasts: bool,
) -> pd.DataFrame:
    """Run holdout test evaluation for pairs with selected tuned parameters."""

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

    train_pairs = set(train_df["pair"].dropna().astype(str).unique())
    test_pairs = set(test_df["pair"].dropna().astype(str).unique())
    shared_pairs = train_pairs.intersection(test_pairs)

    result_rows: list[dict] = []
    skipped = 0

    for _, row in selected_params_df.iterrows():
        pair = str(row["pair"])
        if pair not in shared_pairs:
            skipped += 1
            continue

        p = int(row["selected_p"])
        q = int(row["selected_q"])

        try:
            run = run_arma_for_pair(
                train_df=train_df,
                eval_df=test_df,
                pair=pair,
                spread_col=spread_col,
                p=p,
                q=q,
                entry_z=entry_z,
                exit_z=exit_z,
                min_train_points=min_train_points,
                min_eval_points=min_eval_points,
                eval_split="test",
                window_label=holdout_window,
                forecast_mode="once",
            )
        except Exception:
            run = None

        if run is None:
            skipped += 1
            continue

        forecast_df, metrics, model_summary = run

        out_row = {
            "pair": pair,
            "window_label": holdout_window,
            "eval_split": "test",
            "selected_p": p,
            "selected_q": q,
            "rmse": metrics["rmse"],
            "mae": metrics["mae"],
            "n_train": metrics["n_train"],
            "n_eval": metrics["n_eval"],
            "spread_col": spread_col,
            "mean_val_rmse": row["mean_val_rmse"],
            "mean_val_mae": row["mean_val_mae"],
            "n_tuning_windows": row["n_tuning_windows"],
        }
        result_rows.append(out_row)

        if save_forecasts:
            save_pair_outputs(
                output_root=output_root,
                window_label=holdout_window,
                pair=pair,
                eval_split="holdout",
                forecast_df=forecast_df,
                metrics=out_row,
                fitted_model_summary=model_summary,
            )

    if skipped > 0:
        warnings.warn(f"Skipped {skipped} pairs during holdout evaluation.", stacklevel=2)

    return pd.DataFrame(result_rows)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Generate final holdout ARMA evaluation from tuning-selected parameters."
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
    parser.add_argument("--entry_z", type=float, default=2.0)
    parser.add_argument("--exit_z", type=float, default=0.5)
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=10)
    parser.add_argument("--save_forecasts", action="store_true")
    args = parser.parse_args()

    tuning_root = Path(args.tuning_root)
    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    validation_df = load_validation_tuning_results(
        tuning_root=tuning_root,
        spread_col=args.spread_col,
    )
    selected_params_df = select_final_params_per_pair(
        validation_df=validation_df,
        spread_col=args.spread_col,
    )
    if selected_params_df.empty:
        raise ValueError("No selected parameters were produced from tuning results.")

    final_holdout_df = evaluate_holdout_with_selected_params(
        selected_params_df=selected_params_df,
        input_root=input_root,
        holdout_window=args.holdout_window,
        spread_col=args.spread_col,
        entry_z=args.entry_z,
        exit_z=args.exit_z,
        min_train_points=args.min_train_points,
        min_eval_points=args.min_eval_points,
        output_root=output_root,
        save_forecasts=args.save_forecasts,
    )

    selected_params_path = output_root / "selected_holdout_params.csv"
    final_results_path = output_root / "final_holdout_results.csv"
    selected_params_df.to_csv(selected_params_path, index=False)
    final_holdout_df.to_csv(final_results_path, index=False)

    n_selected = len(selected_params_df)
    n_evaluated = len(final_holdout_df)
    n_skipped = max(0, n_selected - n_evaluated)

    print("\n=== ARMA Holdout Evaluation Complete ===")
    print(f"Tuning root: {tuning_root}")
    print(f"Pair dataset root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Holdout window: {args.holdout_window}")
    print(f"Spread column: {args.spread_col}")
    print(f"Pairs with selected params: {n_selected}")
    print(f"Pairs evaluated on holdout: {n_evaluated}")
    print(f"Pairs skipped: {n_skipped}")
    print(f"Selected params file: {selected_params_path}")
    print(f"Final holdout results file: {final_results_path}")


if __name__ == "__main__":
    main()
