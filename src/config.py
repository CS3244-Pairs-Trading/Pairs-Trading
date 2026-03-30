from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class TimeWindow:
    """Represents a simple inclusive time window."""

    start: str
    end: str


@dataclass(frozen=True)
class ProjectConfig:
    """Central project configuration for data paths and early-pipeline parameters."""

    project_root: Path
    raw_stocks_dir: Path
    data_dir: Path
    interim_dir: Path
    processed_dir: Path
    eda_output_dir: Path

    top_liquid_stocks_path: Path
    selected_stocks_dir: Path

    cleaned_prices_path: Path
    engineered_features_path: Path

    top_n_stocks: int
    liquidity_start_date: str | None
    liquidity_end_date: str | None

    train_window: TimeWindow
    val_window: TimeWindow
    test_window: TimeWindow
    embargo_days: int


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = PROJECT_ROOT / "data"
INTERIM_DIR = DATA_DIR / "interim"
PROCESSED_DIR = DATA_DIR / "processed"
EDA_OUTPUT_DIR = PROJECT_ROOT / "outputs" / "eda"

RAW_STOCKS_DIR = DATA_DIR / "raw" 
TOP_LIQUID_STOCKS_PATH = INTERIM_DIR / "top_1000_liquid_stocks.csv"
SELECTED_STOCKS_DIR = INTERIM_DIR / "top_1000_stocks"

CLEANED_PRICES_PATH = INTERIM_DIR / "prices_clean.csv"
ENGINEERED_FEATURES_PATH = PROCESSED_DIR / "prices_features.csv"

DEFAULT_CONFIG = ProjectConfig(
    project_root=PROJECT_ROOT,
    raw_stocks_dir=RAW_STOCKS_DIR,
    data_dir=DATA_DIR,
    interim_dir=INTERIM_DIR,
    processed_dir=PROCESSED_DIR,
    eda_output_dir=EDA_OUTPUT_DIR,
    top_liquid_stocks_path=TOP_LIQUID_STOCKS_PATH,
    selected_stocks_dir=SELECTED_STOCKS_DIR,
    cleaned_prices_path=CLEANED_PRICES_PATH,
    engineered_features_path=ENGINEERED_FEATURES_PATH,
    top_n_stocks=1000,
    liquidity_start_date=None,
    liquidity_end_date=None,
    train_window=TimeWindow(start="2010-01-01", end="2015-12-31"),
    val_window=TimeWindow(start="2016-01-01", end="2016-12-31"),
    test_window=TimeWindow(start="2017-01-01", end="2017-12-31"),
    embargo_days=0,
)


def ensure_directories(config: ProjectConfig = DEFAULT_CONFIG) -> None:
    """Create output directories used by the early pipeline."""

    for directory in [
        config.data_dir,
        config.interim_dir,
        config.processed_dir,
        config.eda_output_dir,
        config.selected_stocks_dir,
    ]:
        directory.mkdir(parents=True, exist_ok=True)
