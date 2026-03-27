from __future__ import annotations
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_CONFIG, ProjectConfig, ensure_directories


def compute_top_liquid_stocks(
    raw_stocks_dir: Path,
    output_csv_path: Path,
    top_n: int,
    start_date: str | None = None,
    end_date: str | None = None,
) -> pd.DataFrame:
    """Compute and save top-N stocks ranked by average daily dollar volume."""

    rows: list[dict[str, float | str]] = []

    for file_path in sorted(raw_stocks_dir.glob("*.txt")):
        try:
            df = pd.read_csv(file_path)
        except Exception as exc:
            print(f"Warning: skipping {file_path.name} (read error: {exc})")
            continue

        required = {"Date", "Close", "Volume"}
        if not required.issubset(df.columns):
            print(f"Warning: skipping {file_path.name} (missing required columns)")
            continue

        df = df[["Date", "Close", "Volume"]].copy()
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df["Close"] = pd.to_numeric(df["Close"], errors="coerce")
        df["Volume"] = pd.to_numeric(df["Volume"], errors="coerce")

        if start_date is not None:
            df = df[df["Date"] >= pd.Timestamp(start_date)]
        if end_date is not None:
            df = df[df["Date"] <= pd.Timestamp(end_date)]

        df = df.dropna(subset=["Date", "Close", "Volume"])
        if df.empty:
            continue

        avg_dollar_volume = (df["Close"] * df["Volume"]).mean()
        ticker = file_path.stem.split(".")[0]
        rows.append({"Ticker": ticker, "AverageDailyDollarVolume": float(avg_dollar_volume)})

    out = pd.DataFrame(rows)
    if out.empty:
        out = pd.DataFrame(columns=["Ticker", "AverageDailyDollarVolume"])
    else:
        out = out.sort_values("AverageDailyDollarVolume", ascending=False).head(top_n)

    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(output_csv_path, index=False)
    return out


def main(config: ProjectConfig = DEFAULT_CONFIG) -> pd.DataFrame:
    """Run top-liquidity filtering using config paths."""

    ensure_directories(config)
    result = compute_top_liquid_stocks(
        raw_stocks_dir=config.raw_stocks_dir,
        output_csv_path=config.top_liquid_stocks_path,
        top_n=config.top_n_stocks,
        start_date=config.liquidity_start_date,
        end_date=config.liquidity_end_date,
    )

    print(f"Saved top {len(result)} stocks to: {config.top_liquid_stocks_path}")
    return result


if __name__ == "__main__":
    main()
