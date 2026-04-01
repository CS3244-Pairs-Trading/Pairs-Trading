from __future__ import annotations
import argparse
import warnings
from pathlib import Path
import pandas as pd
from src.config import DEFAULT_CONFIG, ProjectConfig, all_training_windows

REQUIRED_COLUMNS = {
    "pair",
    "training_window",
    "is_eligible",
    "score",
    "rank",
    "initial_beta",
}


def load_discovered_pairs(path: Path) -> pd.DataFrame:
    """Load and validate the discovered pairs table."""

    if not path.exists():
        raise FileNotFoundError(
            f"Discovered pairs file not found: {path}. "
            "Run pair ranking first to create discovered_pairs.csv."
        )

    df = pd.read_csv(path)
    missing = REQUIRED_COLUMNS - set(df.columns)
    if missing:
        missing_cols = ", ".join(sorted(missing))
        raise ValueError(
            f"Discovered pairs file is missing required columns: {missing_cols}. "
            f"Path: {path}"
        )

    before = len(df)
    df = df.drop_duplicates().copy()
    dropped = before - len(df)
    if dropped > 0:
        warnings.warn(f"Dropped {dropped} duplicate discovered pair rows.", stacklevel=2)

    return df


def parse_pair_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Split pair symbols into stock_a and stock_b columns."""

    out = df.copy()
    split = out["pair"].astype(str).str.split("-", n=1, expand=True)

    if split.shape[1] < 2:
        raise ValueError("Column 'pair' is not parseable. Expected format like 'aapl-msft'.")

    out["stock_a"] = split[0].str.strip()
    out["stock_b"] = split[1].str.strip()

    invalid = (out["stock_a"] == "") | (out["stock_b"] == "") | out["stock_b"].isna()
    if invalid.any():
        bad_examples = out.loc[invalid, "pair"].head(5).tolist()
        raise ValueError(
            "Found invalid 'pair' values. Expected format like 'aapl-msft'. "
            f"Examples: {bad_examples}"
        )

    return out


def filter_selected_pairs(
    pairs_df: pd.DataFrame,
    training_window: str,
    eligible_only: bool = True,
    top_k: int | None = None,
    sort_by: str = "score",
    ascending: bool = False,
    min_score: float | None = None,
    max_score: float | None = None,
    min_rank: int | None = None,
    max_rank: int | None = None,
) -> pd.DataFrame:
    """Filter and rank selected pairs for one training window."""

    if sort_by not in pairs_df.columns:
        raise ValueError(f"sort_by='{sort_by}' is not a valid column in discovered pairs.")
    if top_k is not None and top_k <= 0:
        raise ValueError("top_k must be > 0 when provided.")

    window_df = pairs_df.loc[pairs_df["training_window"] == training_window].copy()

    if eligible_only:
        window_df = window_df.loc[window_df["is_eligible"] == True]  # noqa: E712
    if min_score is not None:
        window_df = window_df.loc[window_df["score"] >= min_score]
    if max_score is not None:
        window_df = window_df.loc[window_df["score"] <= max_score]
    if min_rank is not None:
        window_df = window_df.loc[window_df["rank"] >= min_rank]
    if max_rank is not None:
        window_df = window_df.loc[window_df["rank"] <= max_rank]

    window_df = window_df.sort_values(sort_by, ascending=ascending)

    if top_k is not None:
        window_df = window_df.head(top_k)

    window_df = parse_pair_columns(window_df).reset_index(drop=True)

    base_cols = [
        "pair",
        "stock_a",
        "stock_b",
        "training_window",
        "is_eligible",
        "score",
        "rank",
        "initial_beta",
    ]
    remaining_cols = [c for c in window_df.columns if c not in base_cols]
    return window_df[base_cols + remaining_cols]


def save_selected_pairs(df: pd.DataFrame, output_path: Path) -> None:
    """Save selected pairs CSV to disk."""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_path, index=False)


def build_selected_pairs_for_all_windows(
    pairs_df: pd.DataFrame | None = None,
    output_dir: Path | None = None,
    top_k: int | None = None,
    eligible_only: bool = True,
    sort_by: str = "score",
    ascending: bool = False,
    config: ProjectConfig = DEFAULT_CONFIG,
    min_score: float | None = None,
    max_score: float | None = None,
    min_rank: int | None = None,
    max_rank: int | None = None,
) -> dict[str, pd.DataFrame]:
    """Build and persist selected pairs for every configured training window."""

    discovered_path = config.processed_dir / "discovered_pairs.csv"
    pairs = pairs_df if pairs_df is not None else load_discovered_pairs(discovered_path)
    out_root = output_dir if output_dir is not None else config.processed_dir / "selected_pairs"

    selected_by_window: dict[str, pd.DataFrame] = {}
    for _, _, label in all_training_windows(config):
        selected_df = filter_selected_pairs(
            pairs_df=pairs,
            training_window=label,
            eligible_only=eligible_only,
            top_k=top_k,
            sort_by=sort_by,
            ascending=ascending,
            min_score=min_score,
            max_score=max_score,
            min_rank=min_rank,
            max_rank=max_rank,
        )
        output_path = out_root / label / "selected_pairs.csv"
        save_selected_pairs(selected_df, output_path)
        if selected_df.empty:
            warnings.warn(f"No selected pairs for window '{label}'.", stacklevel=2)
        selected_by_window[label] = selected_df

    return selected_by_window


def summarize_selected_pairs(selected_by_window: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Return summary counts per window."""

    rows = []
    for label, df in selected_by_window.items():
        eligible_count = int(df["is_eligible"].sum()) if "is_eligible" in df.columns else 0
        rows.append(
            {
                "training_window": label,
                "selected_pairs": len(df),
                "eligible_pairs": eligible_count,
            }
        )
    return pd.DataFrame(rows).sort_values("training_window").reset_index(drop=True)


def main() -> None:
    """CLI entry point for pair selection materialization."""

    parser = argparse.ArgumentParser(
        description="Build selected_pairs.csv files per window from discovered_pairs.csv."
    )
    parser.add_argument("--top_k", type=int, default=None, help="Keep only top K pairs per window.")
    parser.add_argument(
        "--eligible_only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Keep only is_eligible=True pairs.",
    )
    parser.add_argument(
        "--sort_by",
        type=str,
        default="score",
        help="Column used for sorting before top_k selection.",
    )
    parser.add_argument(
        "--ascending",
        action="store_true",
        help="Sort ascending (default is descending).",
    )
    parser.add_argument(
        "--window",
        type=str,
        default=None,
        help="Optional single window label (e.g. 2010_2012).",
    )
    args = parser.parse_args()

    pairs_df = load_discovered_pairs(DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv")
    output_root = DEFAULT_CONFIG.processed_dir / "selected_pairs"

    if args.window:
        valid_labels = {label for _, _, label in all_training_windows(DEFAULT_CONFIG)}
        if args.window not in valid_labels:
            valid = ", ".join(sorted(valid_labels))
            raise ValueError(f"Unknown window '{args.window}'. Valid windows: {valid}")

        selected_df = filter_selected_pairs(
            pairs_df=pairs_df,
            training_window=args.window,
            eligible_only=args.eligible_only,
            top_k=args.top_k,
            sort_by=args.sort_by,
            ascending=args.ascending,
        )
        output_path = output_root / args.window / "selected_pairs.csv"
        save_selected_pairs(selected_df, output_path)
        if selected_df.empty:
            warnings.warn(f"No selected pairs for window '{args.window}'.", stacklevel=2)

        print("Selected pairs build complete.")
        print(f"Window: {args.window}")
        print(f"Rows saved: {len(selected_df)}")
        print(f"Output: {output_path}")
        return

    selected = build_selected_pairs_for_all_windows(
        pairs_df=pairs_df,
        output_dir=output_root,
        top_k=args.top_k,
        eligible_only=args.eligible_only,
        sort_by=args.sort_by,
        ascending=args.ascending,
    )
    summary = summarize_selected_pairs(selected)
    total_rows = int(summary["selected_pairs"].sum()) if not summary.empty else 0

    print("Selected pairs build complete.")
    print(f"Discovered pairs source: {DEFAULT_CONFIG.processed_dir / 'discovered_pairs.csv'}")
    print(f"Output root: {output_root}")
    print(f"Windows processed: {len(summary)}")
    print(f"Total selected rows saved: {total_rows}")
    if not summary.empty:
        print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
