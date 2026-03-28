"""
Reads PCA coordinates produced by pca.py and runs K-Means for K = 20, 30, 40, 50 across all 5 training windows (expanding window approach).
For each window, outputs are saved to the same subfolder as the PCA coordinates.

Output structure (per window)
data/clustering/2010_2012/
    stock_clusters_all_k.csv      --> Each row is a stock and which cluster it was assigned to under each k
    stock_clusters_best_k{K}.csv  --> Same as above but only for the one optimal k
    silhouette_summary.csv        --> Silhouette scores per k
    silhouette_summary.png        --> Silhouette bar chart
    clusters_2d_k{K}.png          --> A scatter plot of all stocks in 2D space using the first two PCA components as axes.
...
"""


from __future__ import annotations

import argparse
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score

from src.config import DEFAULT_CONFIG

warnings.filterwarnings("ignore")


# training windows must match what pca.py produced
TRAINING_WINDOWS = [
    ("2010-01-01", "2012-12-31", "2010_2012"),
    ("2010-01-01", "2013-12-31", "2010_2013"),
    ("2010-01-01", "2014-12-31", "2010_2014"),
    ("2010-01-01", "2015-12-31", "2010_2015"),
    ("2010-01-01", "2016-12-31", "2010_2016"),
]

# 50 colours for scatter plots
_PALETTE = (list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[::8][:40])

def cluster_color(label: int) -> str:
    return _PALETTE[label % len(_PALETTE)]

##################
# for silhouette #
##################

def compute_silhouette(coords: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    sample = min(2000, len(labels))
    return float(silhouette_score(coords, labels, sample_size = sample, random_state = 42))

############
# plotting #
############

def plot_clusters_2d(
    coords: np.ndarray,
    labels: np.ndarray,
    tickers: np.ndarray,
    k: int,
    sil: float,
    pca_variance: list[float],
    window_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize = (12, 8))

    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            c = cluster_color(lbl),
            s = 35, alpha = 0.75, edgecolors = "none",
            label = f"C{lbl}",
        )

    # annotate a random subset to keep the plot readable
    rng = np.random.default_rng(0)
    sample = rng.choice(len(tickers), size = min(60, len(tickers)), replace = False)
    for i in sample:
        ax.annotate(
            tickers[i],
            (coords[i, 0], coords[i, 1]),
            fontsize = 5, alpha = 0.65,
            xytext = (2, 2), textcoords = "offset points",
        )

    ev = pca_variance
    ax.set_xlabel(f"PC1 ({ev[0]:.1%} var)")
    ax.set_ylabel(f"PC2 ({ev[1]:.1%} var)")
    ax.set_title(
        f"K-Means ({window_label.replace('_', '–')},  K = {k},  Silhouette = {sil:.4f})"
    )

    if k <= 20:
        ax.legend(fontsize = 6, ncol = 2, loc = "upper right", markerscale = 1.2)

    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Cluster plot saved: {out_path}")

def plot_evaluation(
    eval_df: pd.DataFrame,
    window_label: str,
    out_path: Path,
) -> None:
    fig, ax = plt.subplots(figsize = (7, 4))
    ks = eval_df.index.astype(str).tolist()

    ax.bar(ks, eval_df["Silhouette"], color = "#4c72b0", width = 0.5)
    ax.set_title(
        f"Silhouette Score by K ({window_label.replace('_', '–')})",
        fontweight = "bold",
    )
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    for i, v in enumerate(eval_df["Silhouette"]):
        if not np.isnan(v):
            ax.text(i, v + 0.001, f"{v:.4f}", ha = "center", va = "bottom", fontsize = 9)

    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Silhouette plot: {out_path}")

####################
# kmeans functions #
####################

def run_kmeans_suite(
    coords: np.ndarray,
    k_values: list[int],
) -> dict[int, np.ndarray]:
    """Fit K-Means for each K. Returns {k: label_array}."""
    results = {}
    n_stocks = len(coords)

    for k in k_values:
        if k >= n_stocks:
            print(f"K = {k}  skipped (only {n_stocks} stocks available)")
            continue
        km = KMeans(n_clusters = k, random_state = 42, n_init = 10, max_iter = 500)
        labels = km.fit_predict(coords)
        results[k] = labels
        print(f"K = {k:<3} done")

    return results


def pick_best_k(eval_df: pd.DataFrame) -> int | None:
    """Pick the K with the highest silhouette score."""
    valid = eval_df.dropna()
    if valid.empty:
        return None
    return int(valid["Silhouette"].idxmax())


def run_kmeans_for_window(
    window_dir: Path,
    window_label: str,
    k_values: list[int],
) -> None:
    """Run K-Means for a single training window."""

    pca_file = window_dir / "pca_coordinates.csv"
    if not pca_file.exists():
        print(f"Skip. PCA file not found: {pca_file}.")
        return

    # 1. load PCA coordinates
    pca_df = pd.read_csv(pca_file, index_col = "Ticker")
    coords = pca_df.values
    tickers = pca_df.index.values
    print(f"Stocks: {len(tickers)}, Components: {pca_df.shape[1]}")

    # approximate per-component variance for axis labels
    comp_var = coords.var(axis = 0)
    pca_variance = (comp_var / comp_var.sum()).tolist()

    # 2. run K-Means
    print(f"Running K-Means for K = {k_values} ...")
    labels_dict = run_kmeans_suite(coords, k_values)

    if not labels_dict:
        print("No valid K values. Skip evaluation.")
        return

    # 3. evaluate with silhouette score
    print("Evaluating ...")
    rows = []
    for k, labels in labels_dict.items():
        sil = compute_silhouette(coords, labels)
        print(f"K={k:<3}  Silhouette = {sil:.4f}")
        rows.append({"K": k, "Silhouette": sil})

    eval_df = pd.DataFrame(rows).set_index("K")
    eval_df.to_csv(window_dir / "silhouette_summary.csv")
    plot_evaluation(eval_df, window_label, window_dir / "silhouette_summary.png")

    # 4. cluster assignments + 2-D plots
    assignment_cols = {"Ticker": tickers}
    for k, labels in labels_dict.items():
        assignment_cols[f"Cluster_K{k}"] = labels
        sil = eval_df.loc[k, "Silhouette"] if k in eval_df.index else float("nan")
        plot_clusters_2d(
            coords, labels, tickers, k, sil,
            pca_variance[:2], window_label,
            window_dir / f"clusters_2d_k{k}.png",
        )

    all_df = pd.DataFrame(assignment_cols)
    all_df.to_csv(window_dir / "stock_clusters_all_k.csv", index = False)
    print(f"All assignments: {window_dir / 'stock_clusters_all_k.csv'}")

    # 5. best K
    best_k = pick_best_k(eval_df)
    if best_k is not None:
        best_df = pd.DataFrame({"Ticker": tickers, "Cluster": labels_dict[best_k]})
        best_path = window_dir / f"stock_clusters_best_k{best_k}.csv"
        best_df.to_csv(best_path, index=False)
        print(f"Best K = {best_k} (silhouette = {eval_df.loc[best_k,'Silhouette']:.4f})")
        print(f"Best assignments: {best_path}")
    else:
        print("Could not choose the optimal k.")

########
# main #
########

def main() -> None:
    parser = argparse.ArgumentParser(
        description = "K-Means clustering on PCA coordinates for all training windows."
    )
    parser.add_argument(
        "--clustering_dir",
        type=Path,
        default = DEFAULT_CONFIG.data_dir / "clustering",
        help = "Root clustering directory (contains one subfolder per window).",
    )
    parser.add_argument(
        "--k_values",
        type = int,
        nargs = "+",
        default = [20, 30, 40, 50],
        help = "K values to evaluate (default: 20 30 40 50).",
    )
    args = parser.parse_args()

    for start_date, end_date, window_label in TRAINING_WINDOWS:
        window_dir = args.clustering_dir / window_label
        print(f"Window {window_label.replace('_', '–')}")
        run_kmeans_for_window(window_dir, window_label, args.k_values)
        print()

    print("Done.")


if __name__ == "__main__":
    main()