from __future__ import annotations
import argparse
import re
import warnings
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories
from src.data_prep.feature_engineering import compute_labels, compute_pair_features


REQUIRED_SELECTED_PAIR_COLUMNS = {
    "pair",
    "stock_a",
    "stock_b",
    "training_window",
    "initial_beta",
}

REQUIRED_PRICE_COLUMNS = {
    "Date",
    "Ticker",
    "Close",
    "SimpleReturn",
    "LogPrice",
}


def load_selected_pairs(path: Path) -> pd.DataFrame:
    """Read and validate a selected pairs CSV."""

    if not path.exists():
        raise FileNotFoundError(f"Selected pairs file not found: {path}")

    df = pd.read_csv(path)
    missing = REQUIRED_SELECTED_PAIR_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"Selected pairs file is missing required columns: {missing_cols}. Path: {path}"
        )

    return df.drop_duplicates().copy()


def load_engineered_prices(path: Path) -> pd.DataFrame:
    """Read engineered long-format prices and validate schema."""

    if not path.exists():
        raise FileNotFoundError(f"Engineered prices file not found: {path}")

    df = pd.read_csv(path, parse_dates=["Date"])
    missing = REQUIRED_PRICE_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"Engineered prices file is missing required columns: {missing_cols}. Path: {path}"
        )

    return df.sort_values(["Date", "Ticker"]).copy()


def get_window_bounds(label: str, config: ProjectConfig = DEFAULT_CONFIG) -> tuple[str, str]:
    """Return the configured train-start and train-end dates for a window label."""

    for fold in config.expanding_folds:
        if fold.label == label:
            return fold.train.start, fold.train.end

    holdout = config.holdout_split
    if holdout.label == label:
        return holdout.train.start, holdout.train.end

    valid = [fold.label for fold in config.expanding_folds] + [holdout.label]
    raise ValueError(f"Unknown window label '{label}'. Valid labels: {', '.join(valid)}")


def _window_specs(config: ProjectConfig) -> list[dict[str, str]]:
    """Return standardized window specs for expanding folds and holdout."""

    specs: list[dict[str, str]] = []
    for fold in config.expanding_folds:
        specs.append(
            {
                "label": fold.label,
                "train_start": fold.train.start,
                "train_end": fold.train.end,
                "eval_start": fold.val.start,
                "eval_end": fold.val.end,
                "eval_name": "val",
                "build_end": fold.val.end,
            }
        )

    holdout = config.holdout_split
    specs.append(
        {
            "label": holdout.label,
            "train_start": holdout.train.start,
            "train_end": holdout.train.end,
            "eval_start": holdout.test.start,
            "eval_end": holdout.test.end,
            "eval_name": "test",
            "build_end": holdout.test.end,
        }
    )
    return specs


def split_pair_dataset_by_dates(
    df: pd.DataFrame,
    train_end: str,
    eval_start: str,
    eval_end: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Split a full pair dataset into train and evaluation partitions by configured dates."""

    if df.empty:
        return df.copy(), df.copy()

    out = df.copy()
    out["Date"] = pd.to_datetime(out["Date"])
    out = out.sort_values(["pair", "Date"]).reset_index(drop=True)

    train_df = out.loc[out["Date"] <= pd.Timestamp(train_end)].copy()
    eval_df = out.loc[
        (out["Date"] >= pd.Timestamp(eval_start)) & (out["Date"] <= pd.Timestamp(eval_end))
    ].copy()

    train_df = train_df.sort_values(["pair", "Date"]).reset_index(drop=True)
    eval_df = eval_df.sort_values(["pair", "Date"]).reset_index(drop=True)
    return train_df, eval_df


def _to_pair_safe_filename(pair: str) -> str:
    """Sanitize a pair name for filesystem-safe output filenames."""

    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", pair.strip())
    return cleaned.strip("._") or "pair"


def _build_raw_pair_frame(window_df: pd.DataFrame, stock_a: str, stock_b: str) -> pd.DataFrame:
    """Build aligned raw columns for two stocks on common dates."""

    keep_cols = ["Date", "Close", "SimpleReturn", "LogPrice"]
    if "Volume" in window_df.columns:
        keep_cols.append("Volume")

    a_df = (
        window_df.loc[window_df["Ticker"] == stock_a, keep_cols]
        .drop_duplicates(subset=["Date"])
        .rename(
            columns={
                "Close": "close_a",
                "SimpleReturn": "return_a",
                "LogPrice": "log_price_a",
                "Volume": "volume_a",
            }
        )
    )
    b_df = (
        window_df.loc[window_df["Ticker"] == stock_b, keep_cols]
        .drop_duplicates(subset=["Date"])
        .rename(
            columns={
                "Close": "close_b",
                "SimpleReturn": "return_b",
                "LogPrice": "log_price_b",
                "Volume": "volume_b",
            }
        )
    )

    return a_df.merge(b_df, on="Date", how="inner").sort_values("Date").reset_index(drop=True)


def build_pair_dataset(
    full_df: pd.DataFrame,
    pair_row: pd.Series,
    start_date: str,
    end_date: str,
    kalman_delta: float = 1e-4,
    label_horizons: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Build one model-ready per-pair dataset for a given date interval."""

    pair = str(pair_row["pair"])
    stock_a = str(pair_row["stock_a"])
    stock_b = str(pair_row["stock_b"])
    training_window = str(pair_row.get("training_window", ""))

    try:
        ols_beta = float(pair_row["initial_beta"])
    except (TypeError, ValueError):
        warnings.warn(f"Skipping pair {pair}: invalid initial_beta.", stacklevel=2)
        return pd.DataFrame()

    date_mask = (
        (full_df["Date"] >= pd.Timestamp(start_date))
        & (full_df["Date"] <= pd.Timestamp(end_date))
    )
    window_df = full_df.loc[date_mask].copy()
    if window_df.empty:
        return pd.DataFrame()

    if stock_a not in set(window_df["Ticker"]) or stock_b not in set(window_df["Ticker"]):
        warnings.warn(
            f"Skipping pair {pair}: missing ticker(s) in engineered prices for this window.",
            stacklevel=2,
        )
        return pd.DataFrame()

    raw = _build_raw_pair_frame(window_df, stock_a=stock_a, stock_b=stock_b)
    if raw.empty:
        warnings.warn(f"Skipping pair {pair}: no aligned common dates.", stacklevel=2)
        return pd.DataFrame()

    raw = raw.set_index("Date").sort_index()
    if len(raw) < 60:
        warnings.warn(f"Pair {pair} has only {len(raw)} aligned observations (<60).", stacklevel=2)

    vol_a = raw["volume_a"] if "volume_a" in raw.columns else None
    vol_b = raw["volume_b"] if "volume_b" in raw.columns else None

    features = compute_pair_features(
        log_price_a=raw["log_price_a"],
        log_price_b=raw["log_price_b"],
        return_a=raw["return_a"],
        return_b=raw["return_b"],
        ols_beta=ols_beta,
        volume_a=vol_a,
        volume_b=vol_b,
        kalman_delta=kalman_delta,
    )
    # kalman_col = features["spread_kalman"] if "spread_kalman" in features.columns else None
    # labels = compute_labels(features["spread_ols"], spread_kalman=kalman_col, horizons=label_horizons)
    labels = compute_labels(
        lp_a=raw["log_price_a"],
        lp_b=raw["log_price_b"],
        current_beta=features["kalman_beta"],
        ols_beta=ols_beta,
        horizons=label_horizons
    )

    pair_df = pd.concat([raw, features, labels], axis=1)
    pair_df = pair_df.loc[:, ~pair_df.columns.duplicated()]
    pair_df["pair"] = pair
    pair_df["stock_a"] = stock_a
    pair_df["stock_b"] = stock_b
    pair_df["training_window"] = training_window
    pair_df = pair_df.reset_index().sort_values("Date").reset_index(drop=True)
    return pair_df


def build_window_dataset(
    full_df: pd.DataFrame,
    selected_pairs_df: pd.DataFrame,
    window_label: str,
    start_date: str,
    end_date: str,
    kalman_delta: float = 1e-4,
    label_horizons: tuple[int, ...] = (5, 10),
) -> pd.DataFrame:
    """Build stacked pair dataset for one window over [start_date, end_date]."""

    window_pairs = selected_pairs_df.loc[
        selected_pairs_df["training_window"] == window_label
    ].copy()
    if window_pairs.empty:
        return pd.DataFrame()

    datasets: list[pd.DataFrame] = []
    for _, pair_row in window_pairs.iterrows():
        pair_df = build_pair_dataset(
            full_df=full_df,
            pair_row=pair_row,
            start_date=start_date,
            end_date=end_date,
            kalman_delta=kalman_delta,
            label_horizons=label_horizons,
        )
        if not pair_df.empty:
            datasets.append(pair_df)

    if not datasets:
        return pd.DataFrame()

    return (
        pd.concat(datasets, ignore_index=True)
        .sort_values(["pair", "Date"])
        .reset_index(drop=True)
    )


def save_window_dataset(df: pd.DataFrame, output_path: Path) -> None:
    """Save a single dataset CSV."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def save_split_pair_datasets(
    full_df: pd.DataFrame,
    train_df: pd.DataFrame,
    eval_df: pd.DataFrame,
    window_dir: Path,
    eval_name: str,
) -> None:
    """Save full/train/eval pair datasets for one window."""

    window_dir.mkdir(parents=True, exist_ok=True)
    full_df.to_csv(window_dir / "pair_dataset.csv", index=False)
    train_df.to_csv(window_dir / "train_pair_dataset.csv", index=False)
    eval_df.to_csv(window_dir / f"{eval_name}_pair_dataset.csv", index=False)


def _save_per_pair_files(window_df: pd.DataFrame, window_dir: Path) -> None:
    """Save optional per-pair CSVs from a full window dataset."""

    if window_df.empty:
        return

    pairs_dir = window_dir / "pairs"
    pairs_dir.mkdir(parents=True, exist_ok=True)
    for pair, pair_df in window_df.groupby("pair", sort=True):
        pair_path = pairs_dir / f"{_to_pair_safe_filename(str(pair))}.csv"
        pair_df.to_csv(pair_path, index=False)


def build_all_pair_datasets(
    full_df: pd.DataFrame | None = None,
    selected_pairs_root: Path | None = None,
    output_root: Path | None = None,
    kalman_delta: float = 1e-4,
    label_horizons: tuple[int, ...] = (5, 10),
    save_per_pair: bool = False,
    config: ProjectConfig = DEFAULT_CONFIG,
    target_window: str | None = None,
) -> dict[str, pd.DataFrame]:
    """
    Build full and split pair datasets for all windows (or one target window).

    Saved per window:
    - pair_dataset.csv
    - train_pair_dataset.csv
    - val_pair_dataset.csv OR test_pair_dataset.csv
    """

    if full_df is None:
        full_df = load_engineered_prices(config.engineered_features_path)

    pairs_root = selected_pairs_root or (config.processed_dir / "selected_pairs")
    out_root = output_root or (config.processed_dir / "pair_datasets")

    specs = _window_specs(config)
    if target_window is not None:
        specs = [s for s in specs if s["label"] == target_window]
        if not specs:
            valid = ", ".join(sorted([s["label"] for s in _window_specs(config)]))
            raise ValueError(f"Unknown window '{target_window}'. Valid windows: {valid}")

    built: dict[str, pd.DataFrame] = {}
    for spec in specs:
        label = spec["label"]
        selected_path = pairs_root / label / "selected_pairs.csv"
        if not selected_path.exists():
            warnings.warn(f"Selected pairs file missing for window '{label}': {selected_path}", stacklevel=2)
            built[label] = pd.DataFrame()
            continue

        selected_pairs_df = load_selected_pairs(selected_path)
        full_window_df = build_window_dataset(
            full_df=full_df,
            selected_pairs_df=selected_pairs_df,
            window_label=label,
            start_date=spec["train_start"],
            end_date=spec["build_end"],
            kalman_delta=kalman_delta,
            label_horizons=label_horizons,
        )

        train_df, eval_df = split_pair_dataset_by_dates(
            df=full_window_df,
            train_end=spec["train_end"],
            eval_start=spec["eval_start"],
            eval_end=spec["eval_end"],
        )

        window_dir = out_root / label
        save_split_pair_datasets(
            full_df=full_window_df,
            train_df=train_df,
            eval_df=eval_df,
            window_dir=window_dir,
            eval_name=spec["eval_name"],
        )

        if save_per_pair:
            _save_per_pair_files(full_window_df, window_dir)

        built[label] = full_window_df
        unique_pairs = int(full_window_df["pair"].nunique()) if not full_window_df.empty else 0
        print(f"Window: {label}")
        print(f"  Full rows: {len(full_window_df)}")
        print(f"  Train rows: {len(train_df)}")
        print(f"  {spec['eval_name'].capitalize()} rows: {len(eval_df)}")
        print(f"  Unique pairs: {unique_pairs}")
        print(f"  Output: {window_dir}")

    return built


def summarize_window_builds(results: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Summarize row/pair counts by window."""

    rows = []
    for label, df in results.items():
        n_rows = len(df)
        n_pairs = int(df["pair"].nunique()) if not df.empty and "pair" in df.columns else 0
        rows.append({"training_window": label, "rows": n_rows, "pairs_built": n_pairs})
    return pd.DataFrame(rows).sort_values("training_window").reset_index(drop=True)


def main() -> None:
    """CLI entry point for pair dataset construction and window-based splitting."""

    parser = argparse.ArgumentParser(
        description="Build full and split pair datasets (train/val/test) from selected pairs."
    )
    parser.add_argument("--window", type=str, default=None, help="Optional single window label.")
    parser.add_argument("--delta", type=float, default=1e-4, help="Kalman delta parameter.")
    parser.add_argument(
        "--horizon",
        type=int,
        nargs="+",
        default=[5, 10],
        help="Label horizons, e.g. --horizon 5 10",
    )
    parser.add_argument(
        "--save_per_pair",
        action="store_true",
        help="Also save one full CSV per pair under pair_datasets/<window>/pairs/",
    )
    args = parser.parse_args()

    ensure_directories(DEFAULT_CONFIG)
    full_df = load_engineered_prices(DEFAULT_CONFIG.engineered_features_path)
    selected_pairs_root = DEFAULT_CONFIG.processed_dir / "selected_pairs"
    output_root = DEFAULT_CONFIG.processed_dir / "pair_datasets"
    horizons = tuple(args.horizon)

    results = build_all_pair_datasets(
        full_df=full_df,
        selected_pairs_root=selected_pairs_root,
        output_root=output_root,
        kalman_delta=args.delta,
        label_horizons=horizons,
        save_per_pair=args.save_per_pair,
        config=DEFAULT_CONFIG,
        target_window=args.window,
    )

    summary = summarize_window_builds(results)
    total_rows = int(summary["rows"].sum()) if not summary.empty else 0
    total_pairs = int(summary["pairs_built"].sum()) if not summary.empty else 0

    print("\nPair dataset build complete.")
    print(f"Engineered prices source: {DEFAULT_CONFIG.engineered_features_path}")
    print(f"Selected pairs root: {selected_pairs_root}")
    print(f"Output root: {output_root}")
    print(f"Windows processed: {len(summary)}")
    print(f"Total rows saved (full datasets): {total_rows}")
    print(f"Total pairs modeled (window-level unique sums): {total_pairs}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
