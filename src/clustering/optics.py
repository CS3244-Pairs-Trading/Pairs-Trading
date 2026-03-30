"""
Output for each training window:
    optics_reachability.png --> bar chart of reachability distances in OPTICS order
    optics_2d.png --> PCA scatter plot coloured by cluster (grey = noise)
    optics_clusters.csv --> Ticker, Cluster (-1 = noise) for every stock
    optics_summary.csv --> min_samples, xi, cluster count, noise count, silhouette
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import OPTICS
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_CONFIG, all_training_windows

# min_samples controls how dense a region must be to start a cluster
# 5 is the standard recommendation from the original OPTICS paper
MIN_SAMPLES = 5

# xi is the minimum steepness of a drop in the reachability plot to count as a cluster boundary. 
# Lower values = more clusters extracted; 0.05 is a safe default.
XI = 0.05

# colours for the clusters
_PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[::8][:40]

def cluster_color(label): # noise points (label == -1) are coloured grey
    return "#bbbbbb" if label == -1 else _PALETTE[label % len(_PALETTE)]

# plot the bar chart of reachability distances in OPTICS traversal order
# valleys in this plot correspond to dense regions (clusters), whereas peaks are the boundaries between them
# grey bars are noise points
def plot_reachability(reachability, ordering, labels, window_label, out_path):
    fig, ax = plt.subplots(figsize = (14, 5))
    for i, idx in enumerate(ordering):
        ax.bar(i, reachability[idx], color=cluster_color(labels[idx]), width = 1.0, alpha = 0.8)
    ax.set_xlabel("Points (OPTICS ordering)")
    ax.set_ylabel("Reachability distance")
    ax.set_title(f"OPTICS Reachability Plot ({window_label.replace('_', '–')})  valleys = clusters, grey = noise")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)

def plot_clusters_2d(coords, labels, tickers, n_clusters, n_noise, sil, window_label, out_path):
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
    ax.set_title(f"OPTICS ({window_label.replace('_', '–')}) "
                 f"clusters = {n_clusters}, noise = {n_noise}, silhouette = {sil_str}")
    
    # legend only for <=20 clusters otherwise it gets cluttered
    if n_clusters <= 20:
        ax.legend(fontsize = 6, ncol = 2, loc = "upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)

# run optics for each window
def run_window(window_dir, window_label):
    pca_file = window_dir / "pca_coordinates.csv"
    if not pca_file.exists():
        print(f"Skip. {pca_file} not found.")
        return

    pca_df = pd.read_csv(pca_file, index_col = "Ticker")
    coords = pca_df.values
    tickers = pca_df.index.values
    print(f"Stocks: {len(tickers)}, Components: {pca_df.shape[1]}")
    print(f"min_samples={MIN_SAMPLES}, xi = {XI}")

    opt = OPTICS(min_samples = MIN_SAMPLES, xi = XI, cluster_method = "xi", metric = "euclidean")
    opt.fit(coords)

    labels = opt.labels_
    n_clusters = len(set(labels) - {-1})
    n_noise = int((labels == -1).sum())
    print(f"Clusters: {n_clusters}, Noise: {n_noise} ({n_noise/len(labels):.1%})")

    # silhouette is only meaningful on actual cluster members, so exclude noise.
    # also need at least 2 clusters and enough non-noise points to compute it
    non_noise = labels != -1
    if n_clusters >= 2 and non_noise.sum() > n_clusters:
        sil = float(silhouette_score(coords[non_noise], labels[non_noise],
                                     sample_size = min(2000, non_noise.sum()), random_state = 42))
    else:
        sil = float("nan")
    print(f"Silhouette: {sil:.4f}" if not np.isnan(sil) else "Silhouette: n/a")

    pd.DataFrame({"Ticker": tickers, "Cluster": labels}).to_csv(
        window_dir / "optics_clusters.csv", index = False)

    pd.DataFrame([{"Window": window_label, "MinSamples": MIN_SAMPLES, "Xi": XI,
                   "NClusters": n_clusters, "NNoise": n_noise,
                   "Silhouette": round(sil, 4) if not np.isnan(sil) else None}]
                 ).to_csv(window_dir / "optics_summary.csv", index = False)

    plot_reachability(opt.reachability_, opt.ordering_, labels,
                      window_label, window_dir / "optics_reachability.png")
    plot_clusters_2d(coords, labels, tickers, n_clusters, n_noise, sil,
                     window_label, window_dir / "optics_2d.png")


if __name__ == "__main__":
    clustering_dir = DEFAULT_CONFIG.data_dir / "clustering"
    for _, _, window_label in all_training_windows:
        print(f"Window {window_label.replace('_', '–')}")
        run_window(clustering_dir / window_label, window_label)
        print()