from __future__ import annotations

import pandas as pd
from src.config import DEFAULT_CONFIG, ExpandingFold, HoldoutSplit, TimeWindow


def filter_dataframe_by_window(
    df: pd.DataFrame,
    window: TimeWindow,
    date_col: str | None = None,
) -> pd.DataFrame:
    """Filter a dataframe to an inclusive date window using DatetimeIndex or a date column."""
    start = pd.Timestamp(window.start)
    end = pd.Timestamp(window.end)

    if date_col is None:
        if not isinstance(df.index, pd.DatetimeIndex):
            raise ValueError("DataFrame index must be DatetimeIndex when date_col is None.")
        return df.loc[(df.index >= start) & (df.index <= end)].copy()

    out = df.copy()
    out[date_col] = pd.to_datetime(out[date_col], errors="coerce")
    return out.loc[(out[date_col] >= start) & (out[date_col] <= end)].copy()


def build_expanding_folds() -> tuple[ExpandingFold, ...]:
    """Return expanding train/validation folds from config."""
    return DEFAULT_CONFIG.expanding_folds


def build_final_holdout_split() -> HoldoutSplit:
    """Return final untouched holdout split from config."""
    return DEFAULT_CONFIG.holdout_split


def materialize_fold_data(
    df: pd.DataFrame,
    fold: ExpandingFold,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, val_df) for one expanding fold."""
    train_df = filter_dataframe_by_window(df, fold.train, date_col=date_col)
    val_df = filter_dataframe_by_window(df, fold.val, date_col=date_col)
    return train_df, val_df


def materialize_holdout_data(
    df: pd.DataFrame,
    holdout_split: HoldoutSplit = DEFAULT_CONFIG.holdout_split,
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for final holdout split."""
    train_df = filter_dataframe_by_window(df, holdout_split.train, date_col=date_col)
    test_df = filter_dataframe_by_window(df, holdout_split.test, date_col=date_col)
    return train_df, test_df


def describe_time_splits() -> dict[str, dict[str, str]]:
    """Return all expanding folds plus final holdout in printable dict form."""
    out: dict[str, dict[str, str]] = {}

    for i, fold in enumerate(DEFAULT_CONFIG.expanding_folds, start=1):
        out[f"fold_{i}_train"] = {"start": fold.train.start, "end": fold.train.end}
        out[f"fold_{i}_val"] = {"start": fold.val.start, "end": fold.val.end}

    out["final_train"] = {
        "start": DEFAULT_CONFIG.holdout_split.train.start,
        "end": DEFAULT_CONFIG.holdout_split.train.end,
    }
    out["final_test"] = {
        "start": DEFAULT_CONFIG.holdout_split.test.start,
        "end": DEFAULT_CONFIG.holdout_split.test.end,
    }

    return out