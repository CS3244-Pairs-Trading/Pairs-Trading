from __future__ import annotations

import pandas as pd

from src.config import DEFAULT_CONFIG, ensure_directories
from src.clustering.pca import run_pca_pipeline
from src.clustering.optics import run_optics_pipeline
from src.pairs_discovery.rank_pairs import run_pair_discovery


def main() -> None:
    """Run the pair selection pipeline end-to-end."""

    config = DEFAULT_CONFIG
    ensure_directories(config)
    clustering_dir = config.data_dir / "clustering"
    clustering_dir.mkdir(parents=True, exist_ok=True)

    print("[1/3] Running PCA on training windows...")
    run_pca_pipeline(config)

    print("[2/3] Running OPTICS clustering...")
    run_optics_pipeline(config)

    print("[3/3] Ranking candidate pairs...")
    discovered_pairs_df = run_pair_discovery(config=config, cluster_source="optics")

    output_path = config.processed_dir / "discovered_pairs.csv"
    total_pairs = len(discovered_pairs_df) if isinstance(discovered_pairs_df, pd.DataFrame) else None

    print("\nPair selection pipeline complete.")
    print(f"Engineered features source: {config.engineered_features_path}")
    print(f"Clustering directory: {clustering_dir}")
    print(f"Discovered pairs file: {output_path}")
    if total_pairs is not None:
        print(f"Total discovered pairs: {total_pairs}")


if __name__ == "__main__":
    main()
