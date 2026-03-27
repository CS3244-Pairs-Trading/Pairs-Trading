from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories


def load_clean_prices(cleaned_prices_path: Path) -> pd.DataFrame:
    """Load cleaned long-format prices."""

    if not cleaned_prices_path.exists():
        raise FileNotFoundError(f"Cleaned dataset not found at {cleaned_prices_path}. Run data_cleaning.py first.")

    df = pd.read_csv(cleaned_prices_path, parse_dates=["Date"])
    return df.sort_values(["Ticker", "Date"]).reset_index(drop=True)


def main(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Create one long-format engineered dataset with SimpleReturn and LogPrice."""

    ensure_directories(config)
    df = load_clean_prices(config.cleaned_prices_path)

    df["SimpleReturn"] = df.groupby("Ticker", sort=False)["Close"].pct_change()
    df["LogPrice"] = np.where(df["Close"] > 0, np.log(df["Close"]), np.nan)

    config.engineered_features_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(config.engineered_features_path, index=False)

    print(f"Engineered rows: {len(df)}")
    print(f"Unique tickers: {df['Ticker'].nunique() if not df.empty else 0}")
    print(f"Saved engineered dataset to: {config.engineered_features_path}")
    return df


if __name__ == "__main__":
    main()
