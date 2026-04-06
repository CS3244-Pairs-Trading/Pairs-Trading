import numpy as np
import pandas as pd
from numba import njit

from src.config import DEFAULT_CONFIG, all_training_windows

from src.pairs_discovery.kalman_hedge import kalman_hedge_ratio

# The OU process in continuous time is: dX_t = κ(θ - X_t)dt + σ dW_t
# Discretised over dt=1 day it becomes an AR(1): X_t = a + b·X_{t-1} + ε_t, ε_t ~ N(0, σ_ε²) where:
#   b = exp(-κ·dt) (AR coefficient)
#   a = θ·(1 - b) (intercept)
#   σ_ε² = σ²·(1 - exp(-2κ·dt)) / (2κ) (innovation variance)
# We recover the OU parameters from OLS estimates of (a, b, σ_ε).

@njit(cache=True)
def _fit_ar1(x: np.ndarray) -> tuple:
    x_prev, x_curr = x[:-1], x[1:]
    n = len(x_prev)
    
    # OLS for AR(1), y = a + bx + error
    mx, my = np.mean(x_prev), np.mean(x_curr) # mean
    num = np.sum((x_prev - mx) * (x_curr - my))
    den = np.sum((x_prev - mx)**2)
    if den == 0: return np.nan, np.nan, np.nan
    
    b = num / den
    a = my - b * mx
    resid_std = np.sqrt(np.sum((x_curr - (a + b * x_prev))**2) / (n - 2)) # error term
    
    return b, a, resid_std

class OUModel:
    # Parameter recovery from AR(1) estimates:
    #  κ = -ln(b) / dt
    #  θ = a / (1 - b)
    #  σ = σ_ε · √(2κ / (1 - b²))
    #  σ_eq = σ / √(2κ) = σ_ε / √(1 - b²)
    # The z-score at time t is: z_t = (X_t - θ) / σ_eq
    def __init__(self, dt: float=1.0):
        self.dt = dt
        self.kappa = np.nan # mean-reversion speed
        self.theta = np.nan # long-run mean
        self.sigma = np.nan # Volatility/diffusion coefficient
        self.eq_std = np.nan # Steady-state standard deviation
        self.b = np.nan # AR(1) coefficient 

    def fit(self, spread_window: np.ndarray) -> bool:
        b, a, resid_std = _fit_ar1(spread_window)
        if np.isnan(b):
            return False
        # Stability check: must be mean-reverting (0 < b < 1)
        if not (0.0 < b < 1.0):
            return False
        
        kappa = -np.log(b) / self.dt
        if abs(kappa) <= 1e-12:
            return False
        theta = a / (1.0 - b)
        # Diffusion coefficient recovery
        sigma = resid_std * np.sqrt(2.0 * kappa / (1.0 - b ** 2))
        # Steady-state variance is sigma^2 / (2 * kappa)
        eq_var = sigma ** 2 / (2.0 * kappa)
        if eq_var <= 0.0:
            return False
        eq_std = np.sqrt(eq_var)
        if not np.isfinite(eq_std) or eq_std < 1e-8:
            return False
        # Update self parameters
        self.b = b
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.eq_std = eq_std
        return True
    
    def predict_next(self, current_value: float, delta: int = 10) -> float:
        # E[Xt+h] = Xt * b^h + theta * (1 - b^h)
        b_h = self.b ** delta
        pred_spread = (current_value * b_h) + (self.theta * (1 - b_h))
        pred_change = pred_spread - current_value
        pred_z = (pred_spread - self.theta) / self.eq_std
        return pred_spread, pred_change, pred_z

    def get_z_score(self, current_value: float) -> float:
        if np.isnan(self.theta) or self.eq_std == 0:
            return np.nan
        return (current_value - self.theta) / self.eq_std
    
    @property
    def half_life(self) -> float:
        if np.isnan(self.kappa) or self.kappa <= 0.0:
            return np.inf
        return np.log(2.0) / self.kappa
    
class OUOrchestrator:
    """
    At each timestep t:
      1. Take the last `lookback` spread values as the training window.
      2. Fit OUModel to that window.
      3. Score the current spread value (outside the window) as a z-score.
    """
    def __init__(self, lookback: int=60, delta: int=10):
        self.lookback = lookback
        self.delta = delta

    def run_walk_forward(self, log_price_df: pd.DataFrame, pairs_metadata: pd.DataFrame, method: str="kalman") -> pd.DataFrame:
        # all_signals = []
        # eligible_pairs = pairs_metadata[pairs_metadata['is_eligible'] == True]
        # available_tickers = set(log_price_df.columns)

        # for _, meta in eligible_pairs.iterrows():
        #     pair_str = str(meta['pair'])
        #     found_tickers = []
        #     for ticker in available_tickers:
        #         # We look for the ticker followed by a hyphen or at the end of the string
        #         if pair_str.startswith(ticker + "-"):
        #             s1 = ticker
        #             s2 = pair_str.replace(ticker + "-", "", 1)
        #             if s2 in available_tickers:
        #                 found_tickers = [s1, s2]
        #                 break
            
        #     if len(found_tickers) != 2:
        #         try:
        #             s1, s2 = pair_str.rsplit('-', 1)
        #         except ValueError:
        #             continue
        #     else:
        #         s1, s2 = found_tickers

        #     if s1 not in available_tickers or s2 not in available_tickers:
        #         continue

        #     # Construct the log-spread
        #     # beta = meta['initial_beta']
        #     # spread = (log_price_df[s1] - beta * log_price_df[s2]).values
        #     dates = log_price_df.index
        #     if method == "kalman":
        #         spread, _ = kalman_spread(log_price_df[s1].values, log_price_df[s2].values, delta=0.01)
        #     else:
        #         beta = meta["initial_beta"]
        #         spread = (log_price_df[s1] - beta * log_price_df[s2]).values
            
        #     model = OUModel()
        #     pair_results = []

        #     for t in range(self.lookback, len(spread)):
        #         window = spread[t - self.lookback : t]
        #         success = model.fit(window)
        #         if not success:
        #             continue
                
        #         z = model.get_z_score(spread[t])
        #         if not np.isfinite(z):
        #             continue
                
        #         pair_results.append({
        #             'date': dates[t],
        #             'pair': meta['pair'],
        #             'z_score': z,
        #             'kappa': model.kappa,
        #             'theta': model.theta,
        #             'sigma': model.sigma,
        #             "eq_std": model.eq_std,
        #             "half_life": model.half_life,
        #         })

        #     all_signals.extend(pair_results)
            
        # return pd.DataFrame(all_signals)
        all_signals = []
        
        for _, row in pairs_metadata.iterrows():
            s1, s2 = row['stock_a'], row['stock_b']
            if s1 not in log_price_df.columns or s2 not in log_price_df.columns:
                continue

            s1_vals = log_price_df[s1].values
            s2_vals = log_price_df[s2].values
            dates = log_price_df.index

            if method == "kalman":
                beta = kalman_hedge_ratio(s1_vals, s2_vals, delta=1e-4)
            else:
                beta = np.full(len(s1_vals), row["initial_beta"])
            
            model = OUModel()
            
            for t in range(self.lookback, len(s1_vals)):
                current_beta = beta[t]
                window_s1 = s1_vals[t - self.lookback : t]
                window_s2 = s2_vals[t - self.lookback : t]
                if method == "kalman":
                    beta_for_window = beta[t - self.lookback] if t - self.lookback >= 0 else beta[t]
                    window = window_s1 - beta_for_window * window_s2
                    current_beta = beta[t]
                else:
                    current_beta = row["initial_beta"]
                    window = window_s1 - current_beta * window_s2
                if not model.fit(window): continue
                
                curr_val = s1_vals[t] - current_beta * s2_vals[t]
                pred_spread, pred_change, pred_z = model.predict_next(curr_val, self.delta)
                
                all_signals.append({
                    'date': dates[t],
                    'pair': row['pair'],
                    'method': method,
                    'z_score': model.get_z_score(curr_val),
                    'pred_spread_10d': pred_spread,
                    'pred_change_10d': pred_change,
                    'pred_z_10d': pred_z,
                    'kappa': model.kappa,
                    'theta': model.theta,
                    'half_life': np.log(2.0) / model.kappa if model.kappa > 0 else np.inf
                })
            
        return pd.DataFrame(all_signals)
    
# def run_ou_model():
#     # Setup files and load data
#     raw_engineered_path = DEFAULT_CONFIG.engineered_features_path
#     raw_df = pd.read_csv(raw_engineered_path, parse_dates=["Date"]) # load raw prices for OU fitting

#     pair_discovery_path = DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv"
#     pairs_df = pd.read_csv(pair_discovery_path)

#     # Prepare data
#     prices_pivot = raw_df.pivot(index="Date", columns="Ticker", values="Close")
#     log_prices_df = np.log(prices_pivot)

#     print(f"\n{'='*50}")
#     print(f"Starting OU Baseline Orchestration")
#     print(f"Targeting: {pairs_df['is_eligible'].sum()} Eligible Pairs")
#     print(f"{'='*50}")

#     orchestrator = OUOrchestrator(lookback=60)
#     all_window_results = []
#     for start_date, end_date, label in all_training_windows(DEFAULT_CONFIG):
#         print("Currently processing window:", label)
#         window_pairs = pairs_df[pairs_df['training_window'] == label]

#         val_start = pd.to_datetime(end_date)
#         val_end = val_start + pd.DateOffset(years=1)
#         hist_start = val_start - pd.DateOffset(days=120) 

#         log_prices_df.index = pd.to_datetime(log_prices_df.index)
#         val_prices = log_prices_df.loc[hist_start : val_end]
#         if val_prices.empty:
#             print(f"  No price data for window {label}, skipping.")
#             continue
#         print(f"  Generating OOS signals ({val_start.date()} → {val_end.date()})...")
#         signals = orchestrator.run_walk_forward(val_prices, window_pairs)
#         oos_signals = signals[signals["date"] >= val_start]
 
#         if not oos_signals.empty:
#             oos_signals = oos_signals.copy()
#             oos_signals["training_window"] = label
#             all_window_results.append(oos_signals)
#             print(f"  {len(oos_signals)} signals generated.")
#         else:
#             print(f"  No OOS signals for window {label}.")
    
#     if all_window_results:
#         final_df = pd.concat(all_window_results, ignore_index=True)
#         final_df = final_df.sort_values('date').drop_duplicates(subset=['date', 'pair'], keep='last')
        
#         output_path = DEFAULT_CONFIG.processed_dir / "ou_baseline_signals.csv"
#         final_df.to_csv(output_path, index=False)
#         print(f"\n{'='*50}")
#         print(f"Total OOS signals: {len(final_df)}")
#         print(f"Pairs covered: {final_df['pair'].nunique()}")
#         print(f"Saved to: {output_path}")
#         print(f"{'='*50}")
#     else:
#         print("\n[FAILED] No valid pairs found for any training window.")
#         return
    
def run_ou_pipeline():
    config = DEFAULT_CONFIG
    raw_df = pd.read_csv(config.engineered_features_path, parse_dates=["Date"])
    prices_pivot = raw_df.pivot(index="Date", columns="Ticker", values="Close")
    log_prices_df = np.log(prices_pivot).sort_index()

    orchestrator = OUOrchestrator(lookback=60, delta=10)
    
    # We run the whole pipeline twice: once for static, once for kalman
    for method in ["static", "kalman"]:
        print(f"\n>>> PROCESSING METHOD: {method.upper()}")
        all_window_results = []

        for start_date, end_date, label in all_training_windows(config):
            pair_path = config.processed_dir / "selected_pairs" / label / "selected_pairs.csv"
            if not pair_path.exists():
                print(f"Skipping {label}: {pair_path} not found.")
                continue
            
            window_pairs = pd.read_csv(pair_path)
            
            val_start = pd.to_datetime(end_date)
            val_end = val_start + pd.DateOffset(years=1)
            hist_start = val_start - pd.DateOffset(days=120) 

            val_prices = log_prices_df.loc[hist_start : val_end]
            if val_prices.empty: continue

            print(f"Window {label}: Generating signals for {len(window_pairs)} pairs...")
            signals = orchestrator.run_walk_forward(val_prices, window_pairs, method=method)
            
            # Filter to only Out-of-Sample (OOS) dates
            oos_signals = signals[signals["date"] >= val_start].copy()
            if not oos_signals.empty:
                oos_signals["training_window"] = label
                all_window_results.append(oos_signals)

        if all_window_results:
            final_df = pd.concat(all_window_results, ignore_index=True)
            output_path = config.processed_dir / f"ou_signals_{method}.csv"
            final_df.to_csv(output_path, index=False)
            print(f"SUCCESS: Saved {len(final_df)} signals to {output_path}")

if __name__ == "__main__":
    run_ou_pipeline()

# if __name__ == "__main__":
#     run_ou_model()