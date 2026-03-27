from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class TimeWindow:
    """Inclusive time window."""

    start: str
    end: str


@dataclass(frozen=True)
class ExpandingFold:
    """One expanding-window validation fold."""

    train: TimeWindow
    val: TimeWindow


def build_expanding_folds() -> list[ExpandingFold]:
    """Return expanding train/validation folds (chronological only)."""

    return [
        ExpandingFold(
            train=TimeWindow("2010-01-01", "2012-12-31"),
            val=TimeWindow("2013-01-01", "2013-12-31"),
        ),
        ExpandingFold(
            train=TimeWindow("2010-01-01", "2013-12-31"),
            val=TimeWindow("2014-01-01", "2014-12-31"),
        ),
        ExpandingFold(
            train=TimeWindow("2010-01-01", "2014-12-31"),
            val=TimeWindow("2015-01-01", "2015-12-31"),
        ),
        ExpandingFold(
            train=TimeWindow("2010-01-01", "2015-12-31"),
            val=TimeWindow("2016-01-01", "2016-12-31"),
        ),
    ]


def build_final_holdout_split() -> tuple[TimeWindow, TimeWindow]:
    """Return final untouched holdout split as (train_window, test_window)."""

    train_window = TimeWindow("2010-01-01", "2016-12-31")
    test_window = TimeWindow("2017-01-01", "2017-12-31")
    return train_window, test_window


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
    holdout_split: tuple[TimeWindow, TimeWindow],
    date_col: str | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Return (train_df, test_df) for final holdout split."""

    train_window, test_window = holdout_split
    train_df = filter_dataframe_by_window(df, train_window, date_col=date_col)
    test_df = filter_dataframe_by_window(df, test_window, date_col=date_col)
    return train_df, test_df


def get_time_splits(*_: Any, **__: Any) -> dict[str, TimeWindow]:
    """Backward-compatible accessor for the final holdout windows."""

    train_window, test_window = build_final_holdout_split()
    return {"train": train_window, "test": test_window}


def describe_time_splits(*_: Any, **__: Any) -> dict[str, dict[str, str]]:
    """Return all expanding folds plus final holdout in printable dict form."""

    out: dict[str, dict[str, str]] = {}

    for i, fold in enumerate(build_expanding_folds(), start=1):
        out[f"fold_{i}_train"] = {"start": fold.train.start, "end": fold.train.end}
        out[f"fold_{i}_val"] = {"start": fold.val.start, "end": fold.val.end}

    final_train, final_test = build_final_holdout_split()
    out["final_train"] = {"start": final_train.start, "end": final_train.end}
    out["final_test"] = {"start": final_test.start, "end": final_test.end}

    return out
