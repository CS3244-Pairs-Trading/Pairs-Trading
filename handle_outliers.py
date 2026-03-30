"""
handle_outliers.py
------------------
Detects and fixes suspicious rows in prices_clean.csv.

Logic:
    A row is flagged as suspicious if:
        |daily_return| > RETURN_THRESHOLD   (extreme price move)
        AND
        volume < VOLUME_MULTIPLIER * ticker_median_volume  (low trading activity)

    Real large moves (GFC, flash crash) attract high volume — they are kept.
    Data errors (bad price entry, unadjusted split) show extreme return but low
    volume — these are masked and forward-filled.

Output:
    Overwrites data/interim/prices_clean.csv with corrected Close prices.
    Returns the cleaned DataFrame for use in the pipeline.

Usage (standalone):
    python src/data_prep/handle_outliers.py

Usage (via pipeline):
    from src.data_prep.handle_outliers import main as run_handle_outliers
    run_handle_outliers(config)
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories


# ── Parameters ────────────────────────────────────────────────────────────────
RETURN_THRESHOLD  = 0.20   # flag rows where |return| exceeds this
VOLUME_MULTIPLIER = 0.10   # flag rows where volume < this * ticker median volume


def _compute_daily_returns(df: pd.DataFrame) -> pd.Series:
    """Compute daily returns per ticker from Close prices."""
    return df.groupby('Ticker')['Close'].pct_change()


def _flag_suspicious(df: pd.DataFrame) -> pd.Series:
    """
    Returns a boolean mask of suspicious rows.

    Suspicious = extreme return AND low volume.
    Genuine extreme moves (high volume) are NOT flagged.
    """
    daily_return = _compute_daily_returns(df)
    median_vol   = df.groupby('Ticker')['Volume'].transform('median')

    extreme_return = daily_return.abs() > RETURN_THRESHOLD
    low_volume     = df['Volume'] < median_vol * VOLUME_MULTIPLIER

    return extreme_return & low_volume


def _apply_fix(df: pd.DataFrame, suspicious: pd.Series) -> pd.DataFrame:
    """
    Masks suspicious Close prices with NaN and forward-fills per ticker.
    Does not modify Volume or any other column.
    """
    df = df.copy()
    df.loc[suspicious, 'Close'] = np.nan
    df['Close'] = df.groupby('Ticker')['Close'].ffill()
    return df


def handle_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """
    Full outlier handling pipeline on a long-format DataFrame.
    Returns a cleaned copy with suspicious Close prices forward-filled.
    """
    suspicious = _flag_suspicious(df)

    n_extreme    = int((_compute_daily_returns(df).abs() > RETURN_THRESHOLD).sum())
    n_suspicious = int(suspicious.sum())
    n_genuine    = n_extreme - n_suspicious

    print(f'  Total extreme rows (|r| > {RETURN_THRESHOLD:.0%})  : {n_extreme:,}')
    print(f'  Suspicious (extreme + low volume)         : {n_suspicious:,}')
    print(f'  Genuine extreme rows (kept as signal)     : {n_genuine:,}')

    if n_suspicious == 0:
        print('  No suspicious rows found — data unchanged.')
        return df

    sample = df[suspicious][['Date', 'Ticker', 'Close', 'Volume']].head(5)
    print(f'\n  Sample of rows being masked and forward-filled:')
    print(sample.to_string(index=False))

    df_clean = _apply_fix(df, suspicious)
    print(f'\n  Fixed {n_suspicious} suspicious rows.')
    return df_clean


def main(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Run outlier handling on prices_clean.csv and overwrite in place."""

    ensure_directories(config)

    if not config.cleaned_prices_path.exists():
        raise FileNotFoundError(
            f'prices_clean.csv not found at {config.cleaned_prices_path}\n'
            'Run data_cleaning.py first.'
        )

    print(f'Loading {config.cleaned_prices_path} ...')
    df = pd.read_csv(config.cleaned_prices_path, parse_dates=['Date'])
    df = df.sort_values(['Ticker', 'Date']).reset_index(drop=True)
    print(f'  Rows: {len(df):,}  |  Tickers: {df["Ticker"].nunique():,}')

    print('\nDetecting and fixing suspicious rows ...')
    df_clean = handle_outliers(df)

    df_clean.to_csv(config.cleaned_prices_path, index=False)
    print(f'\nSaved cleaned data to: {config.cleaned_prices_path}')

    return df_clean


if __name__ == '__main__':
    main()
