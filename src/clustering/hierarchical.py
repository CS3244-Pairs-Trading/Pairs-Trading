"""
Output for each training window:
    hierarchical_dendrogram.png --> dendrogram with the auto-cut threshold mark
    hierarchical_2d.png --> PCA scatter plot coloured by cluster
    hierarchical_clusters.csv --> Ticker, Cluster (0-based) for every stock
    hierarchical_summary.csv --> linkage method, cut threshold, cluster count, silhouette
"""

from pathlib import Path
import matplotlib
matplotlib.use("Agg")  
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import dendrogram, fcluster, linkage
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_CONFIG, all_training_windows

# ward minimises within-cluster variance at each merge step
# it produces compact, similarly-sized clusters, which suits equity data well
LINKAGE_METHOD = "ward"

# colours for the clusters
_PALETTE = list(mcolors.TABLEAU_COLORS.values()) + list(mcolors.CSS4_COLORS.values())[::8][:40]

# to colour the cluster
def cluster_color(label):
    return _PALETTE[label % len(_PALETTE)]

# pick a cut height by finding the largest gap in merge distances (a big jump between consecutive merge distances means
# we are collapsing two genuinely separate groups so we just cut below that jump)
def auto_cut(Z): # the input linkage matrix Z records the distance at which each pair of clusters merged
    gaps = np.diff(Z[:, 2])
    idx = int(np.argmax(gaps))
    # midpoint of the gap so the threshold sits between the two merges
    return float((Z[idx, 2] + Z[idx + 1, 2]) / 2)

def plot_dendrogram(Z, cut_threshold, window_label, out_path):
    fig, ax = plt.subplots(figsize = (14, 5))
    # truncate_mode = "lastp" collapses the bottom of the tree so the plot stays readable
    dendrogram(Z, ax = ax, truncate_mode = "lastp", p = 30,
               leaf_rotation = 90, leaf_font_size = 7, show_contracted = True)
    ax.axhline(cut_threshold, color = "#c44e52", linestyle = "--", linewidth = 1.2,
               label = f"Cut = {cut_threshold:.3f}")
    ax.set_title(f"Hierarchical Clustering Dendrogram ({window_label.replace('_', '–')})")
    ax.set_xlabel("Stock index (or cluster size)")
    ax.set_ylabel("Merge distance")
    ax.legend(fontsize = 9)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)

def plot_clusters_2d(coords, labels, tickers, n_clusters, sil, window_label, out_path):
    fig, ax = plt.subplots(figsize = (12, 8))
    for lbl in np.unique(labels):
        mask = labels == lbl
        ax.scatter(coords[mask, 0], coords[mask, 1], c = cluster_color(lbl - 1),
                   s = 35, alpha = 0.75, edgecolors = "none", label = f"C{lbl}")

    # annotate a random sample of tickers
    rng = np.random.default_rng(0)
    for i in rng.choice(len(tickers), size = min(60, len(tickers)), replace = False):
        ax.annotate(tickers[i], (coords[i, 0], coords[i, 1]),
                    fontsize = 5, alpha = 0.65, xytext = (2, 2), textcoords = "offset points")

    sil_str = f"{sil:.4f}" if not np.isnan(sil) else "n/a"
    ax.set_xlabel("PC1")
    ax.set_ylabel("PC2")
    ax.set_title(f"Hierarchical ({window_label.replace('_', '–')}) "
                 f"clusters = {n_clusters}, silhouette = {sil_str}")
    # legend only for <=20 clusters otherwise it gets cluttered 
    if n_clusters <= 20:
        ax.legend(fontsize = 6, ncol = 2, loc = "upper right")
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)

# run hierarchical clustering for each window
def run_window(window_dir, window_label):
    pca_file = window_dir / "pca_coordinates.csv"
    if not pca_file.exists():
        print(f"Skip. {pca_file} not found.")
        return

    pca_df = pd.read_csv(pca_file, index_col = "Ticker")
    coords = pca_df.values
    tickers = pca_df.index.values
    print(f"Stocks: {len(tickers)}, Components: {pca_df.shape[1]}")

    Z = linkage(coords, method = LINKAGE_METHOD, metric = "euclidean")
    cut_threshold = auto_cut(Z)
    labels = fcluster(Z, t = cut_threshold, criterion = "distance")
    n_clusters = len(np.unique(labels))
    print(f"Cut threshold (auto): {cut_threshold:.4f}")
    print(f"Clusters: {n_clusters}")

    # Silhouette needs at least 2 clusters; cap the sample size to keep it fast
    sil = float(silhouette_score(coords, labels, sample_size = min(2000, len(labels)), random_state = 42)) if n_clusters >= 2 else float("nan")
    print(f"Silhouette: {sil:.4f}" if not np.isnan(sil) else "Silhouette: n/a")

    plot_dendrogram(Z, cut_threshold, window_label, window_dir / "hierarchical_dendrogram.png")

    # cluster returns 1-based labels; shift to 0-based to match OPTICS output convention
    pd.DataFrame({"Ticker": tickers, "Cluster": labels - 1}).to_csv(
        window_dir / "hierarchical_clusters.csv", index = False)

    pd.DataFrame([{"Window": window_label, "Linkage": LINKAGE_METHOD,
                   "CutThreshold": round(cut_threshold, 4), "NClusters": n_clusters,
                   "Silhouette": round(sil, 4) if not np.isnan(sil) else None}]
                 ).to_csv(window_dir / "hierarchical_summary.csv", index = False)

    plot_clusters_2d(coords, labels, tickers, n_clusters, sil,
                     window_label, window_dir / "hierarchical_2d.png")


if __name__ == "__main__":
    clustering_dir = DEFAULT_CONFIG.data_dir / "clustering"
    for _, _, window_label in all_training_windows:
        print(f"Window {window_label.replace('_', '–')}")
        run_window(clustering_dir / window_label, window_label)
        print()