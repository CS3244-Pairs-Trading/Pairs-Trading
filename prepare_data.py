from __future__ import annotations

from src.config import DEFAULT_CONFIG, ensure_directories
from src.data_prep.data_cleaning import main as run_data_cleaning
from src.data_prep.filter_stocks import main as run_filter_stocks
from src.data_prep.isolate_top_1000 import main as run_isolate_stocks
from src.data_prep.handle_outliers import main as run_handle_outliers
from src.data_prep.returns import main as run_feature_engineering
from src.data_prep.splits import describe_time_splits


def main() -> None:
    """Run the data preparation pipeline end-to-end."""

    config = DEFAULT_CONFIG
    ensure_directories(config)

    print("[1/6] Computing top liquid stocks...")
    top_stocks_df = run_filter_stocks(config)

    print("[2/6] Isolating selected stock files...")
    run_isolate_stocks(config)

    print("[3/6] Cleaning selected stock data...")
    clean_df = run_data_cleaning(config)

    print("[4/6] Handling suspicious outliers...")
    clean_df = run_handle_outliers(config)

    print("[5/6] Building one engineered long-format dataset...")
    features_df = run_feature_engineering(config)

    print("[6/6] Time split definitions:")
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
