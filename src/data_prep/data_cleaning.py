from __future__ import annotations

import warnings
from pathlib import Path

import pandas as pd

from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories


RAW_COLUMNS = ["Date", "Open", "High", "Low", "Close", "Volume"]
OUTPUT_COLUMNS = RAW_COLUMNS + ["Ticker"]
PRICE_COLUMNS = ["Open", "High", "Low", "Close"]


def _extract_ticker(file_path: Path) -> str:
    """Extract ticker from filenames like msft.us.txt -> msft."""

    return file_path.stem.split(".")[0].lower()


def _clean_single_stock_file(file_path: Path) -> pd.DataFrame | None:
    """Clean one stock file into the required long-format schema."""

    try:
        df = pd.read_csv(file_path)
    except Exception as exc:
        warnings.warn(f"Skipping {file_path.name}: failed to read ({exc})")
        return None

    missing_columns = [col for col in RAW_COLUMNS if col not in df.columns]
    if missing_columns:
        warnings.warn(f"Skipping {file_path.name}: missing columns {missing_columns}")
        return None

    df = df[RAW_COLUMNS].copy()
    df["Ticker"] = _extract_ticker(file_path)

    df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
    for col in PRICE_COLUMNS + ["Volume"]:
        df[col] = pd.to_numeric(df[col], errors="coerce")

    missing_date_count = int(df["Date"].isna().sum())
    if missing_date_count > 0:
        warnings.warn(f"{file_path.name}: dropping {missing_date_count} rows with missing Date")
    df = df.dropna(subset=["Date"])

    df = df.sort_values("Date")

    before_duplicates = len(df)
    df = df.drop_duplicates(subset=["Date", "Ticker"], keep="last")
    duplicate_removed = before_duplicates - len(df)
    if duplicate_removed > 0:
        warnings.warn(f"{file_path.name}: removed {duplicate_removed} duplicate Date-Ticker rows")

    for col in PRICE_COLUMNS:
        df[col] = df[col].ffill()

    non_positive_price_rows = (df[PRICE_COLUMNS] <= 0).any(axis=1)
    count_non_positive = int(non_positive_price_rows.sum())
    if count_non_positive > 0:
        warnings.warn(f"{file_path.name}: dropping {count_non_positive} rows with non-positive prices")
        df = df.loc[~non_positive_price_rows]

    return df[OUTPUT_COLUMNS]


def clean_selected_stock_data(
    input_dir: Path,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Clean all selected stock files into one long-format dataframe.

    If start_date/end_date are provided, only rows within the range are kept.
    """

    cleaned_frames: list[pd.DataFrame] = []
    for file_path in sorted(input_dir.glob("*.txt")):
        cleaned = _clean_single_stock_file(file_path)
        if cleaned is not None and not cleaned.empty:
            cleaned_frames.append(cleaned)

    if not cleaned_frames:
        return pd.DataFrame(columns=OUTPUT_COLUMNS)

    clean_df = pd.concat(cleaned_frames, ignore_index=True)

    if start_date is not None:
        clean_df = clean_df[clean_df["Date"] >= pd.Timestamp(start_date)]

    if end_date is not None:
        clean_df = clean_df[clean_df["Date"] <= pd.Timestamp(end_date)]

    clean_df = clean_df.sort_values(["Date", "Ticker"]).reset_index(drop=True)

    duplicate_dates_per_ticker = int(clean_df.duplicated(subset=["Ticker", "Date"]).sum())
    if duplicate_dates_per_ticker > 0:
        warnings.warn(
            f"Found {duplicate_dates_per_ticker} duplicate (Ticker, Date) rows after concat; keeping last"
        )
        clean_df = clean_df.drop_duplicates(subset=["Ticker", "Date"], keep="last")

    return clean_df


def main(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Run data cleaning stage for selected top stocks."""

    ensure_directories(config)

    # use the train start and test end from config to define the full analysis window
    start_date = config.analysis_start_date
    end_date = config.analysis_end_date

    clean_df = clean_selected_stock_data(
        config.selected_stocks_dir,
        start_date=start_date,
        end_date=end_date,
    )

    config.cleaned_prices_path.parent.mkdir(parents=True, exist_ok=True)
    clean_df.to_csv(config.cleaned_prices_path, index=False)

    print(f"Date range        : {start_date} to {end_date}")
    print(f"Cleaned rows      : {len(clean_df)}")
    print(f"Unique tickers    : {clean_df['Ticker'].nunique() if not clean_df.empty else 0}")
    print(f"Saved cleaned dataset to: {config.cleaned_prices_path}")
    return clean_df


if __name__ == "__main__":
    main()