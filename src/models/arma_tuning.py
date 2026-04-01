from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG
from src.models.arma import (
    iter_window_dirs,
    load_window_split_datasets,
    run_arma_for_pair,
    save_pair_outputs,
)

P_VALUES = [1, 2, 3, 4, 5]
Q_VALUES = [1, 2, 3, 4, 5]


def _parse_int_list(raw: str | None, default_values: list[int]) -> list[int]:
    if raw is None or raw.strip() == "":
        return default_values
    return [int(x.strip()) for x in raw.split(",") if x.strip() != ""]


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


def _rank_pair_validation_rows(rows: list[dict]) -> list[dict]:
    if not rows:
        return rows
    tmp = pd.DataFrame(rows).sort_values(["rmse", "mae", "p", "q"]).reset_index(drop=True)
    tmp["rank_rmse"] = tmp.index + 1
    return tmp.to_dict(orient="records")


def _concat_train_val(train_df: pd.DataFrame, val_df: pd.DataFrame) -> pd.DataFrame:
    out = pd.concat([train_df, val_df], ignore_index=True)
    out["Date"] = pd.to_datetime(out["Date"], errors="coerce")
    return out.dropna(subset=["Date"]).sort_values(["Date", "pair"]).reset_index(drop=True)


def run_tuning_for_window(
    window_dir: Path,
    output_root: Path,
    spread_col: str,
    p_values: list[int],
    q_values: list[int],
    entry_z: float,
    exit_z: float,
    min_train_points: int,
    min_eval_points: int,
    target_pair: str | None = None,
    save_best_forecasts: bool = True,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Tune ARMA(p, q) using validation RMSE for one window.

    Returns:
    - all_validation_results_df
    - best_validation_results_df
    - test_results_df (possibly empty)
    """

    splits = load_window_split_datasets(window_dir)
    window_label = splits.window_label

    if splits.val is None:
        warnings.warn(
            f"[{window_label}] No validation split found. Skipping tuning for this window.",
            stacklevel=2,
        )
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    train_df = splits.train
    val_df = splits.val
    test_df = splits.test

    pairs = _select_target_pairs(train_df, val_df, pair=target_pair)
    if not pairs:
        warnings.warn(f"[{window_label}] No eligible shared pairs for tuning.", stacklevel=2)
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    print(f"\nWindow: {window_label}")
    print(f"  Candidate pairs for tuning: {len(pairs)}")
    print(f"  Grid size: {len(p_values)} x {len(q_values)}")

    all_val_rows: list[dict] = []
    best_val_rows: list[dict] = []
    test_rows: list[dict] = []

    for idx, pair in enumerate(pairs, start=1):
        print(f"  [{idx}/{len(pairs)}] Tuning pair: {pair}")
        pair_rows: list[dict] = []

        for p in p_values:
            for q in q_values:
                if p == 0 and q == 0:
                    continue
                try:
                    result = run_arma_for_pair(
                        train_df=train_df,
                        eval_df=val_df,
                        pair=pair,
                        spread_col=spread_col,
                        p=p,
                        q=q,
                        entry_z=entry_z,
                        exit_z=exit_z,
                        min_train_points=min_train_points,
                        min_eval_points=min_eval_points,
                        eval_split="val",
                        window_label=window_label,
                        forecast_mode="once",
                    )
                except Exception:
                    continue

                if result is None:
                    continue

                _, metrics, _ = result
                row = {
                    "pair": metrics["pair"],
                    "window_label": metrics["window_label"],
                    "eval_split": "val",
                    "p": metrics["p"],
                    "q": metrics["q"],
                    "rmse": metrics["rmse"],
                    "mae": metrics["mae"],
                    "n_train": metrics["n_train"],
                    "n_eval": metrics["n_eval"],
                    "spread_col": metrics["spread_col"],
                }
                pair_rows.append(row)

        ranked_rows = _rank_pair_validation_rows(pair_rows)
        all_val_rows.extend(ranked_rows)

        if not ranked_rows:
            continue

        best = ranked_rows[0]
        best_val_rows.append(
            {
                "pair": best["pair"],
                "window_label": best["window_label"],
                "best_p": best["p"],
                "best_q": best["q"],
                "rmse": best["rmse"],
                "mae": best["mae"],
                "n_train": best["n_train"],
                "n_eval": best["n_eval"],
                "spread_col": best["spread_col"],
            }
        )

        if save_best_forecasts:
            best_val_run = run_arma_for_pair(
                train_df=train_df,
                eval_df=val_df,
                pair=pair,
                spread_col=spread_col,
                p=int(best["p"]),
                q=int(best["q"]),
                entry_z=entry_z,
                exit_z=exit_z,
                min_train_points=min_train_points,
                min_eval_points=min_eval_points,
                eval_split="val",
                window_label=window_label,
                forecast_mode="once",
            )
            if best_val_run is not None:
                f_df, m_row, m_summary = best_val_run
                save_pair_outputs(
                    output_root=output_root,
                    window_label=window_label,
                    pair=pair,
                    eval_split="val_best",
                    forecast_df=f_df,
                    metrics=m_row,
                    fitted_model_summary=m_summary,
                )

        if test_df is not None:
            train_plus_val = _concat_train_val(train_df, val_df)
            try:
                test_run = run_arma_for_pair(
                    train_df=train_plus_val,
                    eval_df=test_df,
                    pair=pair,
                    spread_col=spread_col,
                    p=int(best["p"]),
                    q=int(best["q"]),
                    entry_z=entry_z,
                    exit_z=exit_z,
                    min_train_points=min_train_points,
                    min_eval_points=min_eval_points,
                    eval_split="test",
                    window_label=window_label,
                    forecast_mode="once",
                )
            except Exception:
                test_run = None

            if test_run is not None:
                f_df, m_row, m_summary = test_run
                test_rows.append(
                    {
                        "pair": m_row["pair"],
                        "window_label": m_row["window_label"],
                        "eval_split": "test",
                        "selected_p": best["p"],
                        "selected_q": best["q"],
                        "rmse": m_row["rmse"],
                        "mae": m_row["mae"],
                        "n_train": m_row["n_train"],
                        "n_eval": m_row["n_eval"],
                        "spread_col": m_row["spread_col"],
                    }
                )
                if save_best_forecasts:
                    save_pair_outputs(
                        output_root=output_root,
                        window_label=window_label,
                        pair=pair,
                        eval_split="test_best",
                        forecast_df=f_df,
                        metrics=m_row,
                        fitted_model_summary=m_summary,
                    )

    all_val_df = pd.DataFrame(all_val_rows)
    best_val_df = pd.DataFrame(best_val_rows)
    test_df_out = pd.DataFrame(test_rows)
    return all_val_df, best_val_df, test_df_out


def _save_window_tuning_outputs(
    output_root: Path,
    window_label: str,
    all_val_df: pd.DataFrame,
    best_val_df: pd.DataFrame,
    test_df: pd.DataFrame,
) -> None:
    window_out = output_root / window_label
    window_out.mkdir(parents=True, exist_ok=True)

    all_val_df.to_csv(window_out / "all_validation_results.csv", index=False)
    best_val_df.to_csv(window_out / "best_validation_results.csv", index=False)
    if not test_df.empty:
        test_df.to_csv(window_out / "test_results.csv", index=False)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Tune ARMA(p, q) per expanding window using validation RMSE."
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
    parser.add_argument("--entry_z", type=float, default=2.0)
    parser.add_argument("--exit_z", type=float, default=0.5)
    parser.add_argument("--min_train_points", type=int, default=30)
    parser.add_argument("--min_eval_points", type=int, default=10)
    parser.add_argument("--p_values", type=str, default=None, help="Comma-separated list, e.g. 1,2,3")
    parser.add_argument("--q_values", type=str, default=None, help="Comma-separated list, e.g. 1,2,3")
    parser.add_argument("--save_best_forecasts", action="store_true")
    args = parser.parse_args()

    p_values = _parse_int_list(args.p_values, P_VALUES)
    q_values = _parse_int_list(args.q_values, Q_VALUES)

    input_root = Path(args.input_root)
    output_root = Path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    window_dirs = iter_window_dirs(input_root)
    if args.window is not None:
        window_dirs = [d for d in window_dirs if d.name == args.window]
        if not window_dirs:
            raise ValueError(f"Window '{args.window}' not found under {input_root}")

    all_validation_frames: list[pd.DataFrame] = []
    best_validation_frames: list[pd.DataFrame] = []
    test_frames: list[pd.DataFrame] = []

    for window_dir in window_dirs:
        try:
            all_val_df, best_val_df, test_df = run_tuning_for_window(
                window_dir=window_dir,
                output_root=output_root,
                spread_col=args.spread_col,
                p_values=p_values,
                q_values=q_values,
                entry_z=args.entry_z,
                exit_z=args.exit_z,
                min_train_points=args.min_train_points,
                min_eval_points=args.min_eval_points,
                target_pair=args.pair,
                save_best_forecasts=args.save_best_forecasts,
            )
        except Exception as exc:
            warnings.warn(f"[{window_dir.name}] Tuning failed: {exc}", stacklevel=2)
            continue

        if all_val_df.empty and best_val_df.empty and test_df.empty:
            continue

        _save_window_tuning_outputs(
            output_root=output_root,
            window_label=window_dir.name,
            all_val_df=all_val_df,
            best_val_df=best_val_df,
            test_df=test_df,
        )

        if not all_val_df.empty:
            all_validation_frames.append(all_val_df)
        if not best_val_df.empty:
            best_validation_frames.append(best_val_df)
        if not test_df.empty:
            test_frames.append(test_df)

    all_validation_results = pd.concat(all_validation_frames, ignore_index=True) if all_validation_frames else pd.DataFrame()
    best_validation_results = pd.concat(best_validation_frames, ignore_index=True) if best_validation_frames else pd.DataFrame()
    test_results = pd.concat(test_frames, ignore_index=True) if test_frames else pd.DataFrame()

    all_validation_results.to_csv(output_root / "all_validation_results.csv", index=False)
    best_validation_results.to_csv(output_root / "best_validation_results.csv", index=False)
    if not test_results.empty:
        test_results.to_csv(output_root / "test_results.csv", index=False)

    print("\n=== ARMA Tuning Complete ===")
    print(f"Input root: {input_root}")
    print(f"Output root: {output_root}")
    print(f"Spread column: {args.spread_col}")
    print(f"Windows attempted: {len(window_dirs)}")
    print(f"Validation rows: {len(all_validation_results)}")
    print(f"Best-validation rows: {len(best_validation_results)}")
    print(f"Test rows: {len(test_results)}")


if __name__ == "__main__":
    main()
