from __future__ import annotations

from dataclasses import replace

from src.config import DEFAULT_CONFIG, ensure_directories
from src.data_prep.data_cleaning import main as run_data_cleaning
from src.data_prep.filter_stocks import main as run_filter_stocks
from src.data_prep.isolate_top_1000 import main as run_isolate_stocks
from src.data_prep.returns import main as run_feature_engineering
from src.data_prep.splits import describe_time_splits


def main() -> None:
    """Run the data preparation pipeline end-to-end (before reducing the time window to 2010-2017 & handling outliers)."""

    config = replace(
        DEFAULT_CONFIG,
        analysis_start_date=None,
        analysis_end_date=None,
        cleaned_prices_path=DEFAULT_CONFIG.cleaned_prices_path.parent / "prices_clean_eda_1.csv",
        engineered_features_path=DEFAULT_CONFIG.engineered_features_path.parent / "prices_features_eda_1.csv"
    )
    ensure_directories(config)

    print("[1/5] Computing top liquid stocks...")
    top_stocks_df = run_filter_stocks(config)

    print("[2/5] Isolating selected stock files...")
    run_isolate_stocks(config)

    print("[3/5] Cleaning selected stock data...")
    clean_df = run_data_cleaning(config)

    print("[4/5] Building one engineered long-format dataset...")
    features_df = run_feature_engineering(config)

    print("[5/5] Time split definitions:")
    split_definitions = describe_time_splits()
    for split_name, bounds in split_definitions.items():
        print(f"  - {split_name}: {bounds['start']} to {bounds['end']}")

    print("\nEarly pipeline complete.")
    print(f"Top stocks count: {len(top_stocks_df)}")
    print(f"Clean rows: {len(clean_df)}")
    print(f"Engineered rows: {len(features_df)}")
    print(f"Final output: {config.engineered_features_path}")


if __name__ == "__main__":
    main()
