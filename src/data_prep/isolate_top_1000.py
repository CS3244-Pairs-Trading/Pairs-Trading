from __future__ import annotations
import shutil
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories


def isolate_selected_stock_files(
    tickers_csv_path: Path,
    source_dir: Path,
    destination_dir: Path,
) -> tuple[list[str], list[str], list[str]]:
    """Copy selected ticker files from source directory to destination directory."""

    top_df = pd.read_csv(tickers_csv_path)
    if "Ticker" not in top_df.columns:
        raise ValueError(f"Expected 'Ticker' column in {tickers_csv_path}")

    requested_tickers = top_df["Ticker"].dropna().astype(str).str.lower().tolist()

    destination_dir.mkdir(parents=True, exist_ok=True)

    copied: list[str] = []
    missing: list[str] = []

    for ticker in requested_tickers:
        filename = f"{ticker}.us.txt"
        source_path = source_dir / filename
        destination_path = destination_dir / filename

        if source_path.exists():
            shutil.copy2(source_path, destination_path)
            copied.append(filename)
        else:
            missing.append(filename)

    return requested_tickers, copied, missing


def main(config: ProjectConfig = DEFAULT_CONFIG) -> tuple[list[str], list[str], list[str]]:
    """Run top-stock file isolation using central config paths."""

    ensure_directories(config)
    requested, copied, missing = isolate_selected_stock_files(
        tickers_csv_path=config.top_liquid_stocks_path,
        source_dir=config.raw_stocks_dir,
        destination_dir=config.selected_stocks_dir,
    )

    print(f"Requested: {len(requested)}")
    print(f"Copied:    {len(copied)}")
    print(f"Missing:   {len(missing)}")
    return requested, copied, missing


if __name__ == "__main__":
    main()
