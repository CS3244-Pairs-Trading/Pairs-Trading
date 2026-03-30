"""
Runs PCA on SimpleReturn for each training window (expanding window approach):
- 2010-2012
- 2010-2013
- 2010-2014
- 2010-2015
- 2010-2016

Output structure:
data/clustering/
    2010_2012/
        pca_coordinates.csv
        pca_variance_returns.png
    2010_2013/
        pca_coordinates.csv
        pca_variance_returns.png
    ...
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from src.config import DEFAULT_CONFIG, all_training_windows

<<<<<<< Updated upstream
from src.config import DEFAULT_CONFIG, all_training_windows
=======
VARIANCE_THRESHOLD = 0.80
>>>>>>> Stashed changes

# HELPER FUNCTIONS

<<<<<<< Updated upstream
'''# training windows: (start_date, end_date, folder_name)
TRAINING_WINDOWS = [
    ("2010-01-01", "2012-12-31", "2010_2012"),
    ("2010-01-01", "2013-12-31", "2010_2013"),
    ("2010-01-01", "2014-12-31", "2010_2014"),
    ("2010-01-01", "2015-12-31", "2010_2015"),
    ("2010-01-01", "2016-12-31", "2010_2016"),
]'''

####################
# helper functions #
####################

def load_return_matrix(
    df: pd.DataFrame,
    start_date: str,
    end_date: str,
    coverage: float = 0.8,
) -> tuple[pd.DataFrame, list[str]]:
    """
    Filter the long-format dataframe to a date window and pivot into a wide return matrix shaped (n_dates, n_tickers).
    Tickers missing more than (1 - coverage) of dates in the window are dropped.
    """
=======
# set up the return matrix
def load_return_matrix(df, start_date, end_date, coverage = 0.8):
    # filter the dataframe into a date window
>>>>>>> Stashed changes
    window = df[(df["Date"] >= pd.Timestamp(start_date)) & (df["Date"] <= pd.Timestamp(end_date))]

    # pivot the long dataframe into a wide return matrix (n_dates, n_tickers)
    matrix = window.pivot(index = "Date", columns = "Ticker", values = "SimpleReturn").sort_index()

    # drop tickers missing more than (1 - coverage) of trading days, then fill remaining NaNs with 0
    matrix = matrix.dropna(axis = 1, thresh = int(len(matrix) * coverage)).fillna(0)
    return matrix, matrix.columns.tolist()

# return the components that cover at least the variance threshold
def select_n_components(explained, threshold):
    n = int(np.searchsorted(np.cumsum(explained), threshold) + 1)
    return max(2, min(n, 15)) # no. of components is capped at 15 to avoid curse of dimensionality

# PLOTTING

# plot the variance
def plot_variance(pca, threshold, n_chosen, out_path, window_label):
    explained = pca.explained_variance_ratio_
    cumulative = np.cumsum(explained)
    idx = np.arange(1, len(explained) + 1)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.bar(idx, explained * 100, color = "#4c72b0", alpha = 0.6, label = "Individual (%)")
    ax.plot(idx, cumulative * 100, color = "#dd8452", linewidth = 2,
            marker = "o", markersize = 3, label = "Cumulative (%)")
    ax.axhline(threshold * 100, color = "#c44e52", linestyle = "--", linewidth = 1.2,
               label = f"Threshold {threshold:.0%}")
    ax.axvline(n_chosen, color = "#8172b2", linestyle = ":", linewidth = 1.2,
               label = f"Chosen n = {n_chosen}")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title(f"PCA on Returns ({window_label.replace('_', '–')})")
    ax.legend(fontsize = 9)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Variance plot saved: {out_path}")


# PCA FUNCTIONS

# run PCA for a each training window and save outputs to output_dir/window_label
def run_pca_for_window(df, start_date, end_date, window_label, output_dir):
    window_dir = output_dir / window_label
    window_dir.mkdir(parents = True, exist_ok = True)

    # 1. build return matrix for this window
    ret_matrix, tickers = load_return_matrix(df, start_date, end_date)
    print(f"Return matrix: {ret_matrix.shape} (dates x tickers)")

    if len(tickers) < 2:
        print("Skip. Not enough tickers after coverage filter.")
        return pd.DataFrame() 

    # 2. standardise
    # PCA expects (n_samples, n_features) transpose so each stock is a sample, each date a feature
    X_scaled = StandardScaler().fit_transform(ret_matrix.T.values)

    # 3. fit on all components first so we can plot the full variance curve and pick n
    pca_full = PCA(random_state=42).fit(X_scaled)
    n_components = select_n_components(pca_full.explained_variance_ratio_, VARIANCE_THRESHOLD)
    cumvar = np.cumsum(pca_full.explained_variance_ratio_)[n_components - 1]
    print(f"Components: {n_components} ({cumvar:.2%} variance covered)")

    plot_variance(pca_full, VARIANCE_THRESHOLD, n_components, window_dir / "pca_variance_returns.png", window_label)

    # 4. refit with only the chosen number of components to get the reduced coordinates
    coords = PCA(n_components=n_components, random_state=42).fit_transform(X_scaled)

    # 5. save coordinates
    col_names = [f"PC{i+1}" for i in range(n_components)]
    pca_df = pd.DataFrame(coords, index = tickers, columns = col_names)
    pca_df.index.name = "Ticker"

    out_csv = window_dir / "pca_coordinates.csv"
    pca_df.to_csv(out_csv)
    print(f"PCA saved: {out_csv}")

<<<<<<< Updated upstream
    return pca_df

def run_all_windows(
    input_file: Path,
    output_dir: Path,
    variance_threshold: float = 0.80,
) -> None:
    # load once, filter per window
    print(f"Loading features from: {input_file}")
    df = pd.read_csv(input_file, parse_dates = ["Date"])
    df = df.dropna(subset = ["SimpleReturn"])
    print(f"Total rows loaded: {len(df)}\n")

    '''for start_date, end_date, window_label in DEFAULT_CONFIG.windows:
        print(f"Window {window_label.replace('_', '–')}")
        run_pca_for_window(
            df, start_date, end_date, window_label,
            output_dir, variance_threshold,
        )
        print()'''
    
    for start_date, end_date, window_label in all_training_windows(DEFAULT_CONFIG):
        print(f"Window {window_label.replace('_', '–')}")
        run_pca_for_window(
            df, start_date, end_date, window_label,
            output_dir, variance_threshold,
        )
        print()

########
# main #
########

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "Run PCA on SimpleReturn for each training window."
    )
    parser.add_argument(
        "--input_file",
        type = Path,
        default = DEFAULT_CONFIG.engineered_features_path,
        help = "Path to prices_features.csv (default: from config).",
    )
    parser.add_argument(
        "--output_dir",
        type = Path,
        default = DEFAULT_CONFIG.data_dir / "clustering",
        help = "Root directory for PCA outputs (one subfolder per window).",
    )
    parser.add_argument(
        "--variance_threshold",
        type = float,
        default = 0.80,
        help = "Cumulative explained-variance target (default: 0.80).",
    )
    args = parser.parse_args()

    run_all_windows(args.input_file, args.output_dir, args.variance_threshold)
    print("Done.")
=======
    return pca_df # returns the PCA coordinates
>>>>>>> Stashed changes


if __name__ == "__main__":
    input_file = DEFAULT_CONFIG.engineered_features_path
    output_dir = DEFAULT_CONFIG.data_dir / "clustering"

    print(f"Loading features from: {input_file}")
    df = pd.read_csv(input_file, parse_dates = ["Date"]).dropna(subset = ["SimpleReturn"])
    print(f"Total rows loaded: {len(df)}\n")

    for start_date, end_date, window_label in all_training_windows:
        print(f"Window {window_label.replace('_', '–')}")
        run_pca_for_window(df, start_date, end_date, window_label, output_dir)
        print()

    print("Done.")