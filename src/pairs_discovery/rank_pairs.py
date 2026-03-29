#!/usr/bin/env python3
# NOTE: Don't forget to run "pip install -r requirements.txt" 

import os
import hashlib
import pickle
from itertools import combinations

import numpy as np
import pandas as pd
from numba import njit
from scipy.stats import spearmanr
from statsmodels.tsa.stattools import coint
from joblib import Parallel, delayed

from pathlib import Path
from src.config import DEFAULT_CONFIG

def get_clusters(pca_df: pd.DataFrame) -> dict: # Based on data/clustering/2010_2012/stock_clusters_best_k20.csv
    return pca_df.groupby("Cluster")["Ticker"].apply(list).to_dict()

def _compute_spread(p1: np.ndarray, p2: np.ndarray) -> np.ndarray:
    # From reference book page 13, spread_t = Y_t - beta * X_t
    # Exact beta will be discovered by Priscilla for spread modeling, this is just for estimation using OLS
    beta = np.polyfit(p2, p1, deg=1)[0]
    return p1 - beta * p2

@njit(cache=True)
def _half_life(spread: np.ndarray) -> float: 
    # This computes how many days for the spread to decay halfway back to its mean
    # From reference book page 9, dy(t) = (lambda * y(t-1) + mu) * dt + de
    # In our case, select dt = 1 for daily observation (1 row per step) and de = 0 for simplicity
    # Selecting de != 0 might affect things since we already have noise on the spread

    # Approximate dy(t) = Δy(t) = y(t) - y(t-1)
    prev = spread[:-1]
    change = spread[1:] - prev

    # Find means of y(t-1) and dy(t)
    prev_mean, change_mean = prev.mean(), change.mean()

    # To find the parameter lambda, minimize the objective function S(λ,μ) = Σ(Δy(t) - (λy(t-1) + μ))^2 using OLS
    slope = (np.sum((prev - prev_mean) * (change - change_mean)) / np.sum((prev - prev_mean) ** 2))

    return np.log(2) / abs(slope) if slope < 0 else np.inf

# def _filter_correlation_score(log_prices: pd.DataFrame, min_corr: float, max_corr: float) -> set:
#     # Identify pairs of assets only for whose correlations fall within a desired range.
#     arr = log_prices.dropna().values
#     cols = log_prices.columns.tolist()
 
#     corr_matrix = np.corrcoef(arr.T) # BLAS to calculate every possible correlation in one shot
#     lower_triangle = np.tri(len(cols), k=-1).T.astype(bool) # since (A,B) and (B,A) are the same, pick one side
#     passes = (corr_matrix >= min_corr) & (corr_matrix <= max_corr) & lower_triangle
 
#     row_idx, col_idx = np.where(passes)
#     return {(cols[r], cols[c]) for r, c in zip(row_idx, col_idx)}

### For spread modeling
@njit(cache=True)
def _hurst_exp(spread: np.ndarray, max_lag: int = 40) -> float: # set to 2 months by default
    # Measure diffusion rate of the spread
    # In a pure random walk (Brownian motion), the distance a price moves increases with the square root of time d(t) ~ t^(0.5)
    # Hurst expontent is to generalize this distance to d(t) ~ t^H where d(t) = |z(T+t)-z(T)|
    # We compute the logarithm instead, so ln(d(t)) = H ln(t) + const, such that it becomes linear in logs
    lags = np.arange(2, max_lag)
    log_lags, log_tau = np.log(lags.astype(np.float64)), np.empty(len(lags))
 
    for i, lag in enumerate(lags):
        diff = spread[lag:] - spread[:-lag]
        log_tau[i] = np.log(np.std(diff) + 1e-12)
 
    lag_mean, tau_mean = log_lags.mean(), log_tau.mean()
    slope = (np.sum((log_lags - lag_mean) * (log_tau - tau_mean)) / np.sum((log_lags - lag_mean) ** 2)) # OLS again
    return slope

# Performance Optimization
class PairCache:
    # Cache key = ticker names and fingerprint of the last 5 rows of input data
    def __init__(self, path: str):
        self.path  = path
        self.store = {}
        if os.path.exists(path):
            with open(path, "rb") as f:
                self.store = pickle.load(f)
 
    def get(self, s1, s2, fingerprint):
        return self.store.get(f"{s1}__{s2}__{fingerprint}")
 
    def set(self, s1, s2, fingerprint, value):
        self.store[f"{s1}__{s2}__{fingerprint}"] = value
 
    def flush(self):
        with open(self.path, "wb") as f:
            pickle.dump(self.store, f)

def data_fingerprint(arr: np.ndarray) -> str:
    return hashlib.md5(arr[-5:].tobytes()).hexdigest()

### Main pair scoring logic
def score_pairs(df: pd.DataFrame) -> pd.Series:
    return (
        0.60 * (1 - df["coint_pval"]) +
        0.40 * (1 - (df["half_life"] - 20).abs() / 40)
    )

def evaluate_pair(s1, s2, cluster_id, p1: np.ndarray, p2: np.ndarray, cache: PairCache, coint_threshold, min_half_life, max_half_life, hurst_max_lag) -> dict | None:
    fingerprint = data_fingerprint(p1) + data_fingerprint(p2)
    cached = cache.get(s1, s2, fingerprint)
    if cached is not None:
        return cached
    
    # Pair evaluation follows from Reference Book Part 3.5, page 33
    is_eligible = True
    pearson  = np.corrcoef(p1, p2)[0, 1]
    spearman = spearmanr(p1, p2).statistic
 
    try: # Engle-Granger cointegration test
        _, eg_pval, _ = coint(p1, p2)
    except Exception:
        return {
            "pair": f"{s1}-{s2}",
            "cluster": cluster_id,
            "coint_pval": np.nan,
            "hurst": np.nan,
            "mean_crossings": np.nan,
            "half_life": np.nan,
            "initial_beta": np.nan,
            "pearson": pearson,
            "spearman": spearman,
            "is_eligible": False,
        }
    if eg_pval > coint_threshold: # if not cointegrated, it's not good for trading
        is_eligible = False
 
    beta = np.polyfit(p2, p1, deg=1)[0]
    spread = _compute_spread(p1, p2)
    
    hl = _half_life(spread)
    if not (min_half_life <= hl <= max_half_life): # if spread takes too long or too short to revert, it's not good for trading
        is_eligible = False

    h_exp = _hurst_exp(spread, max_lag=hurst_max_lag)
    if h_exp > 0.5: # if spread is diffusive, it's not good for mean reversion
        is_eligible = False

    crossings = ((spread[:-1] * spread[1:]) < 0).sum()
    if crossings < 12: # if spread doesn't cross mean enough times, it's not good for trading
        is_eligible = False
 
    result = {
        "pair": f"{s1}-{s2}",
        "cluster": cluster_id,
        "coint_pval": eg_pval,
        "hurst": h_exp,
        "mean_crossings": crossings,
        "half_life": hl,
        "initial_beta": beta,
        "pearson": pearson,
        "spearman": spearman,
        "is_eligible": is_eligible,
    }
 
    cache.set(s1, s2, fingerprint, result)
    return result

def find_candidate_pairs( prices_df: pd.DataFrame, clusters: dict, n_jobs: int = -1, cache_path: str = ".pair_cache.pkl", 
    coint_threshold: float = 0.05, min_half_life: float = 1, max_half_life: float = 252, hurst_max_lag: int = 40) -> pd.DataFrame:
 
    log_prices = np.log(prices_df)
    price_arr, tickers = log_prices.values, log_prices.columns.tolist()
    ticker_pos = {t: i for i, t in enumerate(tickers)}
 
    # corr_passing = _filter_correlation_score(log_prices, min_corr, max_corr)
    cache = PairCache(cache_path)
 
    pairs_to_check = []
    for cluster_id, stocks in clusters.items():
        for s1, s2 in combinations(stocks, 2):
            if s1 not in ticker_pos or s2 not in ticker_pos:
                continue
 
            i1, i2 = ticker_pos[s1], ticker_pos[s2]
 
            both_valid = ~(np.isnan(price_arr[:, i1]) | np.isnan(price_arr[:, i2]))
            p1 = price_arr[both_valid, i1]
            p2 = price_arr[both_valid, i2]
 
            if len(p1) < 252: # need at least one year of data
                continue
 
            pairs_to_check.append((s1, s2, cluster_id, p1, p2))
 
    results = Parallel(n_jobs=n_jobs, backend="loky")(
        delayed(evaluate_pair)(
            s1, s2, cluster_id, p1, p2, cache, coint_threshold,
            min_half_life, max_half_life, hurst_max_lag
        )
        for s1, s2, cluster_id, p1, p2 in pairs_to_check
    )
 
    cache.flush()
 
    df = pd.DataFrame([r for r in results if r is not None])
    if df.empty:
        return df
    
    df["score"] = score_pairs(df)
    df = df.sort_values("score", ascending=False).reset_index(drop=True)
    df["rank"] = range(1, len(df) + 1)
    return df

### Main Execution Logic
def run_pair_discovery():
    raw_engineered_path = DEFAULT_CONFIG.engineered_features_path
    full_df = pd.read_csv(raw_engineered_path, parse_dates=["Date"]) # load raw prices for pair selection

    # Copy windows definition from src/clustering/pca.py
    WINDOWS = [
        ("2010-01-01", "2012-12-31", "2010_2012"),
        ("2010-01-01", "2013-12-31", "2010_2013"),
        ("2010-01-01", "2014-12-31", "2010_2014"),
        ("2010-01-01", "2015-12-31", "2010_2015"),
        ("2010-01-01", "2016-12-31", "2010_2016"),
    ]

    all_window_results = []
    for start_date, end_date, label in WINDOWS:
        print("Currently processing window:", label)
        # Filter and pivot data for this window
        window_mask = (full_df["Date"] >= start_date) & (full_df["Date"] <= end_date)
        window_df = full_df[window_mask]

        if window_df.empty:
            print(f"Warning: No data found for window {label}. Skipping processing.")
            continue

        prices_pivot = window_df.pivot(index="Date", columns="Ticker", values="Close")

        cluster_file = DEFAULT_CONFIG.data_dir / "clustering" / label / "stock_clusters_best_k20.csv"
        cluster_df = pd.read_csv(cluster_file)
        clusters_dict = get_clusters(cluster_df)

        window_results_df = find_candidate_pairs(
            prices_df=prices_pivot,
            clusters=clusters_dict,
            n_jobs=-1,
            coint_threshold=0.05
        )

        if not window_results_df.empty:
            window_results_df["training_window"] = label
            window_results_df["window_pair_id"] = window_results_df["pair"] + "_" + label
            n_eligible = window_results_df["is_eligible"].sum()
            print(f"Done: Screened {len(window_results_df)} pairs. {n_eligible} marked as IS_ELIGIBLE.")
            
            all_window_results.append(window_results_df)
    
    if all_window_results:
        final_df = pd.concat(all_window_results, ignore_index=True)
        final_df = final_df.sort_values(by=["is_eligible", "score"], ascending=[False, False])
        
        output_path = DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n{'='*50}")
        print(f"STATE 3 COMPLETE: FULL AUDIT GENERATED")
        print(f"Total pairs analyzed: {len(final_df)}")
        print(f"Tradeable pairs (is_eligible=True): {final_df['is_eligible'].sum()}")
        print(f"File: {output_path}")
        print(f"{'='*50}")
    else:
        print("\n[FAILED] No eligible pairs found in any window.")

if __name__ == "__main__":
    run_pair_discovery()



# ## Test Script
# def generate_test_data(n_days=500):
#     np.random.seed(42)
#     t = np.arange(n_days)
    
#     # Cointegrated & fast reverting
#     # Passed: Corr high, Coint p < 0.05, Half-life ~10-15
#     x1 = np.cumsum(np.random.normal(0, 0.01, n_days)) + 100
#     noise1 = 0.05 * np.random.normal(0, 1, n_days)

#     for i in range(1, n_days):
#         noise1[i] = 0.85 * noise1[i-1] + np.random.normal(0, 0.01)
#     y1 = 1.2 * x1 + 5 + noise1
    
#     # Correlated but not cointegrated
#     # Passed: Corr high | Failed: Coint p > 0.05
#     x2 = np.cumsum(np.random.normal(0.001, 0.01, n_days)) + 50
#     y2 = np.cumsum(np.random.normal(0.001, 0.01, n_days)) + 50
    
#     # Cointegrated but slow half-life
#     # Passed: Corr high, Coint p < 0.05 | Failed: Half-life > 60
#     x3 = np.cumsum(np.random.normal(0, 0.01, n_days)) + 20
#     noise3 = np.zeros(n_days)
#     for i in range(1, n_days):
#         noise3[i] = 0.995 * noise3[i-1] + np.random.normal(0, 0.01)
#     y3 = 0.5 * x3 + 10 + noise3
    
#     # Pure random walk
#     # Failed: Everything
#     x4 = np.random.normal(100, 5, n_days)
#     y4 = np.random.normal(100, 5, n_days)

#     data = {
#         "GOLD_A": np.exp(x1), "GOLD_B": np.exp(y1),
#         "SPUR_A": np.exp(x2), "SPUR_B": np.exp(y2),
#         "ZOMB_A": np.exp(x3), "ZOMB_B": np.exp(y3),
#         "RAND_A": x4,         "RAND_B": y4
#     }
    
#     df = pd.DataFrame(data)
#     clusters = {
#         "Cluster_1": ["GOLD_A", "GOLD_B", "SPUR_A", "SPUR_B"],
#         "Cluster_2": ["ZOMB_A", "ZOMB_B", "RAND_A", "RAND_B"]
#     }
    
#     return df, clusters

# # Generate and Run
# test_prices, test_clusters = generate_test_data()
# print(test_clusters)
# print(test_prices)
# results = find_candidate_pairs(test_prices, test_clusters)
# print(results)
