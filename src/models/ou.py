import numpy as np
import pandas as pd
from numba import njit

from src.config import DEFAULT_CONFIG, all_training_windows

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
        self.kappa = kappa
        self.theta = theta
        self.sigma = sigma
        self.eq_std = eq_std
        return True

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
    def __init__(self, lookback: int=60):
        self.lookback = lookback

    def run_walk_forward(self, log_price_df: pd.DataFrame, pairs_metadata: pd.DataFrame) -> pd.DataFrame:
        all_signals = []
        eligible_pairs = pairs_metadata[pairs_metadata['is_eligible'] == True]

        for _, meta in eligible_pairs.iterrows():
            s1, s2 = meta['pair'].split('-')
            beta = meta['initial_beta']

            # Construct the log-spread
            spread = (log_price_df[s1] - beta * log_price_df[s2]).values
            dates = log_price_df.index
            
            model = OUModel()
            pair_results = []

            for t in range(self.lookback, len(spread)):
                window = spread[t - self.lookback : t]
                success = model.fit(window)
                if not success:
                    continue
                
                z = model.get_z_score(spread[t])
                if not np.isfinite(z):
                    continue
                
                pair_results.append({
                    'date': dates[t],
                    'pair': meta['pair'],
                    'z_score': z,
                    'kappa': model.kappa,
                    'theta': model.theta,
                    'sigma': model.sigma,
                    "eq_std":    model.eq_std,
                    "half_life": model.half_life,
                })

            all_signals.extend(pair_results)
            
        return pd.DataFrame(all_signals)
    
def run_ou_model():
    # Setup files and load data
    raw_engineered_path = DEFAULT_CONFIG.engineered_features_path
    raw_df = pd.read_csv(raw_engineered_path, parse_dates=["Date"]) # load raw prices for OU fitting

    pair_discovery_path = DEFAULT_CONFIG.processed_dir / "discovered_pairs.csv"
    pairs_df = pd.read_csv(pair_discovery_path)

    # Prepare data
    prices_pivot = raw_df.pivot(index="Date", columns="Ticker", values="Close")
    log_prices_df = np.log(prices_pivot)

    print(f"\n{'='*50}")
    print(f"Starting OU Baseline Orchestration")
    print(f"Targeting: {pairs_df['is_eligible'].sum()} Eligible Pairs")
    print(f"{'='*50}")

    orchestrator = OUOrchestrator(lookback=60)
    all_window_results = []
    for start_date, end_date, label in all_training_windows(DEFAULT_CONFIG):
        print("Currently processing window:", label)
        window_pairs = pairs_df[pairs_df['training_window'] == label]

        val_start = end_date
        val_end = val_start + pd.DateOffset(years=1)
        hist_start = val_start - pd.DateOffset(days=120) 

        val_prices = log_prices_df.loc[hist_start : val_end]
        if val_prices.empty:
            print(f"  No price data for window {label}, skipping.")
            continue
        print(f"  Generating OOS signals ({val_start.date()} → {val_end.date()})...")
        signals = orchestrator.run_walk_forward(val_prices, window_pairs)
        oos_signals = signals[signals["date"] >= val_start]
 
        if not oos_signals.empty:
            oos_signals = oos_signals.copy()
            oos_signals["training_window"] = label
            all_window_results.append(oos_signals)
            print(f"  {len(oos_signals)} signals generated.")
        else:
            print(f"  No OOS signals for window {label}.")
    
    if all_window_results:
        final_df = pd.concat(all_window_results, ignore_index=True)
        final_df = final_df.sort_values('date').drop_duplicates(subset=['date', 'pair'], keep='last')
        
        output_path = DEFAULT_CONFIG.processed_dir / "ou_baseline_signals.csv"
        final_df.to_csv(output_path, index=False)
        print(f"\n{'='*50}")
        print(f"Total OOS signals: {len(final_df)}")
        print(f"Pairs covered: {final_df['pair'].nunique()}")
        print(f"Saved to: {output_path}")
        print(f"{'='*50}")
    else:
        print("\n[FAILED] No valid pairs found for any training window.")
        return

if __name__ == "__main__":
    run_ou_model()