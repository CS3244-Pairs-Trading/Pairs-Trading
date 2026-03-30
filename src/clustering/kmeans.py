"""
Reads PCA coordinates produced by pca.py and runs K-Means for K = 2–30 across all training windows (expanding window approach).
For each window, outputs are saved to the same subfolder as the PCA coordinates.

Output structure (per window)
data/clustering/2010_2012/
    stock_clusters_all_k.csv --> Each row is a stock and which cluster it was assigned to under each k
    stock_clusters_best_k{K}.csv --> Same as above but only for the one optimal k
    silhouette_summary.csv --> Silhouette scores per k
    silhouette_summary.png --> Silhouette bar chart
...
"""

from __future__ import annotations
from pathlib import Path
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from src.config import DEFAULT_CONFIG, all_training_windows

K_VALUES = list(range(2, 31))

# to compute silhoette
def compute_silhouette(coords, labels):
    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(silhouette_score(coords, labels, sample_size = min(2000, len(labels)), random_state = 42))

# PLOTTING
def plot_evaluation(eval_df, window_label, out_path):
    fig, ax = plt.subplots(figsize = (14, 5))  # wider so 29 bars have room to breathe
    ks = eval_df.index.astype(str).tolist()
    ax.bar(ks, eval_df["Silhouette"], color = "#4c72b0", width = 0.6)
    ax.set_title(f"Silhouette Score by K ({window_label.replace('_', '–')})", fontweight = "bold")
    ax.set_xlabel("K")
    ax.set_ylabel("Silhouette Score")
    ax.tick_params(axis = "x", labelsize = 8)
    for i, v in enumerate(eval_df["Silhouette"]):
        if not np.isnan(v):
            ax.text(i, v + 0.001, f"{v:.3f}", ha = "center", va = "bottom",
                    fontsize = 6.5, rotation = 90)
    fig.tight_layout()
    fig.savefig(out_path, dpi = 150)
    plt.close(fig)
    print(f"Silhouette plot: {out_path}")

# K-MEANS FUNCTIONS

def run_kmeans_suite(coords, k_values):
    results = {}
    for k in k_values:
        if k >= len(coords):
            print(f"K = {k}  skipped (only {len(coords)} stocks available)")
            continue
        labels = KMeans(n_clusters = k, random_state = 42, n_init = 10, max_iter = 500).fit_predict(coords)
        results[k] = labels
        print(f"K = {k:<3} done")
    return results

# to choose the K with the highest silhouette (higher means tighter, better-separated clusters)
def pick_best_k(eval_df):
    valid = eval_df.dropna()
    return int(valid["Silhouette"].idxmax()) if not valid.empty else None

# to run k-means for each training window
def run_kmeans_for_window(window_dir, window_label):
    pca_file = window_dir / "pca_coordinates.csv"
    if not pca_file.exists():
        print(f"Skip. PCA file not found: {pca_file}.")
        return

    # 1. load PCA coordinates
    pca_df = pd.read_csv(pca_file, index_col = "Ticker")
    coords = pca_df.values
    tickers = pca_df.index.values
    print(f"Stocks: {len(tickers)}, Components: {pca_df.shape[1]}")

    # 2. run k-means
    print(f"Running K-Means for K = {K_VALUES} ...")
    labels_dict = run_kmeans_suite(coords, K_VALUES)

    if not labels_dict:
        print("No valid K values. Skip evaluation.")
        return

    # 3. evaluate with silhouette score
    print("Evaluating ...")
    rows = []
    for k, labels in labels_dict.items():
        sil = compute_silhouette(coords, labels)
        print(f"K = {k:<3}  Silhouette = {sil:.4f}")
        rows.append({"K": k, "Silhouette": sil})

    eval_df = pd.DataFrame(rows).set_index("K")
    eval_df.to_csv(window_dir / "silhouette_summary.csv")
    plot_evaluation(eval_df, window_label, window_dir / "silhouette_summary.png")

    # 4. cluster assignments
    assignment_cols = {"Ticker": tickers}
    for k, labels in labels_dict.items():
        assignment_cols[f"Cluster_K{k}"] = labels

    # save all k assignments in one file for easy comparison
    all_df = pd.DataFrame(assignment_cols)
    all_df.to_csv(window_dir / "stock_clusters_all_k.csv", index = False)
    print(f"All assignments: {window_dir / 'stock_clusters_all_k.csv'}")

    # 5. choose the best k
    best_k = pick_best_k(eval_df)
    if best_k is not None:
        best_df = pd.DataFrame({"Ticker": tickers, "Cluster": labels_dict[best_k]})
        best_path = window_dir / f"stock_clusters_best_k{best_k}.csv"
        best_df.to_csv(best_path, index = False)
        print(f"Best K = {best_k} (silhouette = {eval_df.loc[best_k,'Silhouette']:.4f})")
        print(f"Best assignments: {best_path}")
    else:
        print("Could not choose the optimal k.")


if __name__ == "__main__":
    clustering_dir = DEFAULT_CONFIG.data_dir / "clustering"
    for _, _, window_label in all_training_windows():
        window_dir = clustering_dir / window_label
        print(f"Window {window_label.replace('_', '–')}")
        run_kmeans_for_window(window_dir, window_label)
        print()
    print("Done.")