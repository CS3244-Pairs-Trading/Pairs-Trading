from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, ProjectConfig, TimeWindow
from src.data_prep.data_cleaning import clean_selected_stock_data
from src.data_prep.filter_stocks import compute_top_liquid_stocks
from src.data_prep.isolate_top_1000 import isolate_selected_stock_files
from src.data_prep.returns import load_clean_prices
from src.data_prep.splits import get_time_splits as _get_time_splits


def load_raw_data(raw_data_dir: Path, pattern: str = "*.txt") -> list[Path]:
    """List raw stock files available for processing."""

    return sorted(raw_data_dir.glob(pattern))


def select_top_stocks(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Select and save top liquid stocks."""

    return compute_top_liquid_stocks(
        raw_stocks_dir=config.raw_stocks_dir,
        output_csv_path=config.top_liquid_stocks_path,
        top_n=config.top_n_stocks,
        start_date=config.liquidity_start_date,
        end_date=config.liquidity_end_date,
    )


def isolate_selected_stocks(config: ProjectConfig = DEFAULT_CONFIG) -> tuple[list[str], list[str], list[str]]:
    """Copy selected stock files to the configured interim folder."""

    return isolate_selected_stock_files(
        tickers_csv_path=config.top_liquid_stocks_path,
        source_dir=config.raw_stocks_dir,
        destination_dir=config.selected_stocks_dir,
    )


def clean_prices(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Clean selected stock files into long-format prices."""

    return clean_selected_stock_data(config.selected_stocks_dir)


def compute_returns(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker simple returns on long-format cleaned data."""

    out = clean_df.sort_values(["Ticker", "Date"]).copy()
    out["SimpleReturn"] = out.groupby("Ticker", sort=False)["Close"].pct_change()
    return out


def compute_log_prices(clean_df: pd.DataFrame) -> pd.DataFrame:
    """Compute per-ticker log prices on long-format cleaned data."""

    out = clean_df.sort_values(["Ticker", "Date"]).copy()
    out["LogPrice"] = np.nan
    positive = out["Close"] > 0
    out.loc[positive, "LogPrice"] = np.log(out.loc[positive, "Close"])
    return out


def load_clean_data(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Load cleaned prices from configured storage path."""

    return load_clean_prices(config.cleaned_prices_path)


def get_time_splits(config: ProjectConfig = DEFAULT_CONFIG) -> dict[str, TimeWindow]:
    """Get train/test windows from split module."""

    return _get_time_splits(config)
