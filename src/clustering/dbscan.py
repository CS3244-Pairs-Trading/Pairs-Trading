"""
Runs DBSCAN on PCA coordinates for each training window.

Output structure (per window):
data/clustering/2010_2012/
    dbscan_clusters.csv --> columns are Ticker, Cluster (-1 = noise/outlier)
    dbscan_summary.csv --> n_clusters, n_noise, silhouette, epsilon
    dbscan_2d.png --> 2-D scatter plot coloured by cluster
    dbscan_kdistance.png --> k-distance plot used to select epsilon
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from sklearn.metrics import silhouette_score
from sklearn.neighbors import NearestNeighbors
from src.config import DEFAULT_CONFIG, all_training_windows

# min_samples: minimum points to form a dense region
MIN_SAMPLES = 5  # book uses min_pts = 5

# 50 colours for the clusters
_PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[::8][:40]

# to colour each cluster, cluster -1 (noise) is grey
def cluster_color(label):
    return "#bbbbbb" if label == -1 else _PALETTE[label % len(_PALETTE)]

# EPSILON SELECTION

# compute the k-distance for each point (distance to its k-th nearest neighbour, where k = min_samples)
def compute_kdistance(coords, min_samples):
    nbrs = NearestNeighbors(n_neighbors = min_samples).fit(coords)
    distances, _ = nbrs.kneighbors(coords)
    return np.sort(distances[:, -1])[::-1] # take the distance to the k-th neighbour (last column)

def pick_epsilon_from_kdistance(kdist):
    """
    to pick an epsilon:
    1. draw a straight line from the first point on the k-distance curve to the last point
    2. for every point on the curve, measure its shortest distance to that line
    3. choose the point with the largest perpendicular distance
    4. that point is the elbow point or epsilon
    """
    n = len(kdist)
    idx = np.arange(n)
    
    # line from first to last point
    p1, p2 = np.array([0, kdist[0]]), np.array([n - 1, kdist[-1]])
    line = p2 - p1

    # perpendicular distance from each point to the line
    perp = np.abs(np.cross(line, p1 - np.column_stack([idx, kdist]))) / np.linalg.norm(line)
    return float(kdist[int(np.argmax(perp))])

# PLOTTING

def plot_kdistance(kdist, epsilon, window_label, out_path):
    fig, ax = plt.subplots(figsize = (8, 4))
    ax.plot(kdist, color = "#4c72b0", linewidth = 1.2)
    ax.axhline(epsilon, color = "#c44e52", linestyle = "--", linewidth = 1.2,
               label = f"Chosen ε = {epsilon:.3f}")
    ax.set_xlabel("Points (sorted by decreasing k-distance)")
    ax.set_ylabel("k-distance")
    ax.set_title(f"k-Distance Plot ({window_label.replace('_', '–')})")
    ax.legend(fontsize = 9)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"k-distance plot: {out_path}")

def plot_clusters_2d(coords, labels, tickers, epsilon, n_clusters, n_noise, sil, window_label, out_path):
    fig, ax = plt.subplots(figsize = (12, 8))
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c = cluster_color(lbl),
                   s = 35, alpha = 0.75, edgecolors = "none",
                   label = "Noise" if lbl == -1 else f"C{lbl}")

    # annotate a random sample of tickers
    rng = np.random.default_rng(0)
    for i in rng.choice(len(tickers), size = min(60, len(tickers)), replace = False):
        ax.annotate(tickers[i], (coords[i, 0], coords[i, 1]),
                    fontsize = 5, alpha = 0.65, xytext = (2, 2), textcoords = "offset points")

    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "n/a"
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"DBSCAN ({window_label.replace('_', '–')})  "
                 f"ε = {epsilon:.3f}, Clusters = {n_clusters}, "
                 f"Noise = {n_noise}, Silhouette = {sil_str}")

    # legend for small cluster counts only
    if n_clusters <= 20:
        ax.legend(fontsize = 6, ncol = 2, loc = "upper right", markerscale = 1.2)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Cluster plot: {out_path}")

# DBSCAN FUNCTIONS

# run dbscan for each training window
def run_dbscan_for_window(window_dir, window_label, epsilon = None):
    pca_file = window_dir / "pca_coordinates.csv"
    if not pca_file.exists():
        print(f"Skip. PCA file not found: {pca_file}.")
        return

    pca_df = pd.read_csv(pca_file, index_col = "Ticker")
    coords = pca_df.values
    tickers = pca_df.index.values
    print(f"Stocks: {len(tickers)}, Components: {pca_df.shape[1]}")

    # use the k-distance plot to find a good epsilon if epsilon hasn't been given
    kdist = compute_kdistance(coords, MIN_SAMPLES)
    if epsilon is None:
        epsilon = pick_epsilon_from_kdistance(kdist)
        print(f"Epsilon (auto): {epsilon:.4f}")
    else:
        print(f"Epsilon (manual): {epsilon:.4f}")

    plot_kdistance(kdist, epsilon, window_label, window_dir / "dbscan_kdistance.png")

    # run DBSCAN
    labels = DBSCAN(eps=epsilon, min_samples = MIN_SAMPLES, metric = "euclidean").fit_predict(coords)
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"Clusters found: {n_clusters}")
    print(f"Noise points: {n_noise} ({n_noise/len(labels):.1%})")

    # silhouette only makes sense if we have at least 2 clusters and some non-noise points
    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        sil = float(silhouette_score(coords[non_noise], labels[non_noise],
                                     sample_size = min(2000, non_noise.sum()), random_state=42))
    else:
        sil = float("nan")
    print(f"Silhouette score: {sil:.4f}" if not np.isnan(sil) else "Silhouette score: n/a")

    pd.DataFrame({"Ticker": tickers, "Cluster": labels}).to_csv(
        window_dir / "dbscan_clusters.csv", index = False)
    print(f"Assignments saved: {window_dir / 'dbscan_clusters.csv'}")

    pd.DataFrame([{"Window": window_label, "Epsilon": round(epsilon, 4),
                   "MinSamples": MIN_SAMPLES, "No. of Clusters": n_clusters, "No. of Noise": n_noise,
                   "Silhouette": round(sil, 4) if not np.isnan(sil) else None}]
                 ).to_csv(window_dir / "dbscan_summary.csv", index = False)

    plot_clusters_2d(coords, labels, tickers, epsilon, n_clusters, n_noise, sil,
                     window_label, window_dir / "dbscan_2d.png")


if __name__ == "__main__":
    clustering_dir = DEFAULT_CONFIG.data_dir / "clustering"
    for _, _, window_label in all_training_windows:
        window_dir = clustering_dir / window_label
        print(f"Window {window_label.replace('_', '–')}")
        run_dbscan_for_window(window_dir, window_label)
        print()

    print("Done.")