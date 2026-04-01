from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class BacktestParamsConfig:
    """
    Market constants and default strategy parameters for backtest_engine.py.

    The z-score fields are defaults used when ZScoreSignal is auto-created
    from BacktestConfig; they have no effect when a custom SignalGenerator
    is passed to engine.run().
    """
    # Market constants
    trading_days:     int   = 252    # trading days per calendar year
    risk_free_rate:   float = 0.0    # annualised risk-free rate (0 = no hurdle)
    fitness_tv_floor: float = 0.125  # turnover floor in Fitness formula (12.5 %)

    # Z-score signal defaults
    entry_z:            float = 2.0    # enter when |z| crosses this
    exit_z:             float = 0.5    # exit  when |z| falls below this
    stop_z:             float = 4.0    # stop-loss at this |z|  (0 = disabled)
    use_rolling_zscore: bool  = False  # False = fixed training stats; True = rolling
    rolling_lookback:   int   = 63     # rolling window in trading days (≈ 3 months)

    # Capital & portfolio
    initial_capital:      float = 1_000_000.0
    n_top_pairs:          int   = 50    # max pairs per training window
    transaction_cost_bps: float = 10.0  # basis points per side per trade leg
    min_pair_score:       float = 0.0   # discard pairs below this composite score
    margin_multiplier: float = 20.0
    output_dir: str = "outputs/backtest"

DEFAULT_BACKTEST_PARAMS = BacktestParamsConfig()


# ──────────────────────────────────────────────────────────
# PIPELINE / TIME-WINDOW CONFIGS  (unchanged below)
# ──────────────────────────────────────────────────────────

@dataclass(frozen=True)
class TimeWindow:
    start: str
    end: str


@dataclass(frozen=True)
class ExpandingFold:
    label: str
    train: TimeWindow
    val: TimeWindow


@dataclass(frozen=True)
class HoldoutSplit:
    label: str
    train: TimeWindow
    test: TimeWindow


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

    analysis_start_date: str | None
    analysis_end_date: str | None
    expanding_folds: tuple[ExpandingFold, ...]
    holdout_split: HoldoutSplit


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
    analysis_start_date=None,
    analysis_end_date=None,
    expanding_folds=(
        ExpandingFold(
            label="2010_2012",
            train=TimeWindow("2010-01-01", "2012-12-31"),
            val=TimeWindow("2013-01-01", "2013-12-31")
        ),
        ExpandingFold(
            label="2010_2013",
            train=TimeWindow("2010-01-01", "2013-12-31"),
            val=TimeWindow("2014-01-01", "2014-12-31")
        ),
        ExpandingFold(
            label="2010_2014",
            train=TimeWindow("2010-01-01", "2014-12-31"),
            val=TimeWindow("2015-01-01", "2015-12-31")
        ),
        ExpandingFold(
            label="2010_2015",
            train=TimeWindow("2010-01-01", "2015-12-31"),
            val=TimeWindow("2016-01-01", "2016-12-31")
        )
    ),
    holdout_split=HoldoutSplit(
        label="2010_2016",
        train=TimeWindow("2010-01-01", "2016-12-31"),
        test=TimeWindow("2017-01-01", "2017-12-31")
    )
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


def all_training_windows(config: ProjectConfig = DEFAULT_CONFIG,) -> tuple[tuple[str, str, str], ...]:
    windows = tuple(
        (fold.train.start, fold.train.end, fold.label)
        for fold in config.expanding_folds
    )

    holdout = config.holdout_split
    holdout_window = (
        holdout.train.start,
        holdout.train.end,
        holdout.label,
    )

    return windows + (holdout_window,)